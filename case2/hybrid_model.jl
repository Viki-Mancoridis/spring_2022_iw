using OrdinaryDiffEq, Flux, Optim, Random, Plots, Plots.PlotMeasures
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load
using Plots: pdf
import FiniteDiff


Random.seed!(1234);

###################################
# Arguments 
is_restart = false; # was true
n_epoch = 10000;
n_plot = 50;
datasize = 50; 
tstep = 1;
n_exp_train = 20; 
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;
noise = 0.05;
ns = 6;
nr = 3;
alg = AutoTsit5(Rosenbrock23(autodiff=false));
atol = 1e-6;
rtol = 1e-3;
species_names = ["TG", "ROH", "DG", "MG", "GL", "R'CO2R"];
rxn_order_names = ["TG", "ROH", "DG", "MG", "GL", "R'CO2R", "Bias"];
current_los_cos = 0;

# opt = ADAMW(5.f-3, (0.9, 0.999), 1.f-6);
# opt = Flux.Optimiser(ExpDecay(5e-3, 0.5, 500 * n_exp_train, 1e-4),
#                      ADAMW(0.005, (0.9, 0.999), 1.f-6));
opt = Flux.Optimiser(ExpDecay(5e-3, 0.5, 500 * n_exp_train, 1e-4),
                     ADAMW(0.0008, (0.9, 0.999), 1.f-6)); 

lb = 1.f-6;
ub = 1.f1;
####################################

function trueODEfunc(dydt, y, k, t)
    # TG(1),ROH(2),DG(3),MG(4),GL(5),R'CO2R(6)
    r1 = k[1] * y[1] * y[2];
    r2 = k[2] * y[3] * y[2];
    r3 = k[3] * y[4] * y[2];
    dydt[1] = - r1;  # TG
    dydt[2] = - r1 - r2 - r3;  # TG
    dydt[3] = r1 - r2;  # DG
    dydt[4] = r2 - r3;  # MG
    dydt[5] = r3;  # GL
    dydt[6] = r1 + r2 + r3;  # R'CO2R
    dydt[7] = 0.f0;
end

logA = Float32[18.60f0, 19.13f0, 7.93f0];
Ea = Float32[14.54f0, 14.42f0, 6.47f0];  # kcal/mol

function Arrhenius(logA, Ea, T)
    R = 1.98720425864083f-3
    k = exp.(logA) .* exp.(-Ea ./ R ./ T)
    return k
end

# Generate datasets
u0_list = rand(Float32, (n_exp, ns + 1));
u0_list[:, 1:2] = u0_list[:, 1:2] .* 2.0 .+ 0.2;
u0_list[:, 3:ns] .= 0.0;
u0_list[:, ns + 1] = u0_list[:, ns + 1] .* 20.0 .+ 323.0;  # T[K]
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);

ode_data_list = zeros(Float32, (n_exp, ns, datasize));
yscale_list = [];
function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end
for i in 1:n_exp
    u0 = u0_list[i, :]
    k = Arrhenius(logA, Ea, u0[end])
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k) # removed , kwargshandle=KeywordArgSilent
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))[1:end - 1, :] # removed , kwargshandle=KeywordArgSilent
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(yscale_list, max_min(ode_data))
end
yscale = maximum(hcat(yscale_list...), dims=2);

np = nr * (ns + 2) + 1;
p = randn(Float32, np) .* 0.1;
p[1:nr] .+= 0.8;
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;
p[end] = 0.1;

function p2vec(p)
    slope = p[nr * (ns + 2) + 1] .* 100
    w_b = p[1:nr] .* slope
    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr)
    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope)
    w_in = clamp.(-w_out, 0, 4)
    w_in = vcat(w_in, w_in_Ea')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("species (column) reaction (row)")
    println("w_in | w_b")
    w_in_ = vcat(w_in, w_b')'
    show(stdout, "text/plain", round.(w_in_, digits=3))
    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end
display_p(p)

inv_R = - 1 / 1.98720425864083f-3;
function crnn(du, u, p, t)
    logX = @. log(clamp(u[1:end - 1], lb, ub))
    w_in_x = w_in' * vcat(logX, inv_R / u[end])
    du .= vcat(w_out * (@. exp(w_in_x + w_b)), 0.f0)
end

u0 = u0_list[1, :];
prob = ODEProblem(crnn, u0, tspan, saveat=tsteps, atol=atol, rtol=rtol) # removed , kwargshandle=KeywordArgSilent

sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());

# explaining predict_neuralode

# The input u0 represents the initial condition of the system, and p represents 
# the parameters of the neural ODE model. p2vec is a function that converts a parameter 
# dictionary into vectors w_in, w_b, and w_out.
# solve(prob, alg, u0=u0, p=p, sensalg=sense) solves the neural ODE model defined by 
# prob using the specified solver algorithm alg, with initial condition u0, and 
# parameters p. The sensalg keyword argument specifies the algorithm to use for 
# computing sensitivities with respect to parameters, which is used for training the model.

# The output of solve is an array of Dual values representing the predicted solution 
# of the system. Array() converts this array to a regular numerical array, and 
# clamp. applies element-wise clipping of the values to the range of [-ub, ub], 
# where ub is a scalar constant.

# The final line returns the clipped prediction pred.
function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p)
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p, sensalg=sense)), -ub, ub) # removed , kwargshandle=KeywordArgSilent
    return pred
end
predict_neuralode(u0, p)

# find number of elements of [C,O,H] from a matrix stoichoiometric
# coefficients (rows) associated with rxns (columns)
# compositions is a matrix that looks like this:
# columns are [C,O,H]
# rows are [TG,ROH,DG,MG,GL,RCOOR] (the different species in the set of rxns)
find_elements = function(w_matrix, compositions)
   elements = zeros(3,3)
   carbon_sum = 0
   oxygen_sum = 0
   hydrogen_sum = 0
   for rxn in 1:size(w_matrix, 2)
      for species in 1:size(w_matrix, 1)
         # global carbon_sum, oxygen_sum, hydrogen_sum
         carbon_sum += w_matrix[species,rxn] * compositions[species,1]
         oxygen_sum += w_matrix[species,rxn] * compositions[species,2]
         hydrogen_sum += w_matrix[species,rxn] * compositions[species,3]
      end

      elements[rxn,1:end] = [carbon_sum, oxygen_sum, hydrogen_sum]
   end

   return elements
end

# define a loss associated with conservation of species
# this loss is wrt to the learned weights of the models, not wrt the
# predicted outputs compared to some "true" output
# model is predicting rates of changes for the concentration of species
loss_cos = function(p)
    w_in_temp, w_b_temp, w_out_temp = p2vec(p)

    compositions = Float32[9 6 14; 1 1 4; 7 5 12; 5 4 10; 3 3 8; 3 2 6]

    # find number of elements going in and out
    elements = find_elements(w_out_temp, compositions)

    # each entry in this matrix should add to 0 here
    # convert this to a loss (MSE)
    squared = elements .^2
    loss = mean(squared) / (n_exp) # scaled down by 50 instead of 300
    return loss
end


# new_loss_compute = function L(constraint_loss_val, my_loss_val)
#     # Compute the value of the combined loss function
#     du0, dp = ForwardDiff.gradient(x -> my_function(x[1:3], x[4]), [u0; p])
#     total_loss_val = my_loss_val + constraint_loss_val
    
#     # Compute the derivatives of the combined loss function
#     my_loss_derivs = ForwardDiff.gradient(my_loss, x)
#     constraint_loss_derivs = ForwardDiff.gradient(constraint_loss, x)
#     total_loss_derivs = my_loss_derivs + constraint_loss_derivs
    
#     # Return the Dual object representing the combined loss function
#     return Dual(total_loss_val, total_loss_derivs)
# end

new_loss_cos = function(p, is_end_of_epoch_loss)
    # println("inside new_loss_cos")
    values = p
    if (!is_end_of_epoch_loss) 
        values = [d.value for d in p] # this works
    end
    p = values # see if this works for end_of_epoch loss or no
    # println("-----------")
    # println("p: ")
    # display(p)
    # println("-----------")

    w_in_temp, w_b_temp, w_out_temp = p2vec(p)

    # println("made it 1")

    # display(w_out_temp)

    compositions = Float32[9 6 14; 1 1 4; 7 5 12; 5 4 10; 3 3 8; 3 2 6]

    # w_out_temp_matrix = w_out_temp
    # if (!is_end_of_epoch_loss)
    #     w_out_temp_matrix = [d.value for d in w_out_temp] # this works
    # end
    # w_out_temp = w_out_temp_matrix

    # find number of elements going in and out
    elements = find_elements(w_out_temp, compositions)

    # println("made it 2")

    # each entry in this matrix should add to 0 here
    # convert this to a loss (MSE)
    squared = elements .^2
    loss = mean(squared) / (n_exp) # scaled down by 50 instead of 300

    # println("made it 3")
    return loss
end

# define a loss associated with conservation of species
# this loss is wrt to the learned weights of the models, not wrt the
# predicted outputs compared to some "true" output
# model is predicting rates of changes for the concentration of species
loss_collision_limit = function(p)
   # println("inside loss_collision_limit")

    w_in_temp, w_b_temp, w_out_temp = p2vec(p)

   #  println("------------")
   #  println("w_b_temp: ")
    
   #  display(w_b_temp)
   #  println("------------")

    w_enforced = Float32[22.57, 18.14, 16.30] # physically enforced

    diff = w_b_temp - w_enforced # predicted rate constants_{i} - enforced rate constants k_{i}

    # println("------------")
    # println("diff: ")
    # display(diff)
    # println("------------")

    loss = sum(max.(diff, 0))

    # println("about to return...")
    # display(loss)

    # println("---------------")

    return loss
end

loss_cos_numerical_gradient = function(p)
    # println("made it here")
    # grad_numerical = ForwardDiff.gradient(p -> new_loss_cos(p, false), p) ==> this gives right data type, but all 0s => didn't work
    # idea: use numerical gradient to approximate gradient at x=p.

    grad_numerical = FiniteDiff.finite_difference_gradient(p -> new_loss_cos(p, false), p) # this actually gives non-zero gradients! :OOO

    # below could still potentially work
    # g_prime = FiniteDiff.gradient(x -> new_loss_cos(p, false), ForwardMode(), mutates = false)
    # output = g_prime(p)

    # grad_numerical = FiniteDiff.finite_difference_gradient(new_loss_cos, p, 0.1);
    # grad_numerical = g(p)
    # fdiff = FiniteDiff(h=0.1)
    # grad_numerical = ForwardDiff.gradient(x -> new_loss_cos(x, false), p, fdiff)
    # grad_numerical = ForwardDiff.gradient(x -> new_loss_cos(x, false), p, ForwardDiff.NumDiff(h=0.1, method="forward")) # added NumDiff()

    values = grad_numerical
    values = [d.value for d in grad_numerical] # this works

    # println("-------------------------------------------------------------")
    # println()
    # println("grad_numerical:") 
    # display(values)
    # println()
    # println("-------------------------------------------------------------")
    # println("-------------------------------------------------------------")
    # println()
    # println("other choie:") 
    # display(grad_numerical[1])
    # println()
    # println("-------------------------------------------------------------")
    return values
end


i_obs = [1, 2, 3, 4, 5, 6];

function loss_neuralode(p, i_exp, is_end_of_epoch_loss)
    global current_los_cos
    ode_data = @view ode_data_list[i_exp, i_obs, :]
    # println("-------------------------------------------------------------")
    # println()
    # println("ODE DATA ./ yscale[i_obs]:")
    # display(ode_data ./ yscale[i_obs]) # ode_data is a 6x50 matrix of data --> makes sense
    # println()
    # println("-------------------------------------------------------------")
    # println("this is the p passed into predict_neuralode: ")
    # display(p)
    # println()
    # println("------------")
    
    pred = predict_neuralode(u0_list[i_exp, :], p)[i_obs, :] # solve the system using the given p
    # println("-------------------------------------------------------------")
    # println()
    # println("yscale[i_obs]:")
    # display(yscale[i_obs])
    # println()
    # println("-------------------------------------------------------------")

    # println("-------------------------------------------------------------")
    # println()
    # println("p:") # 25-element vector of Dual types
    # values = [d.value for d in p] # this works 
    # display(values)
    # println()
    # println("-------------------------------------------------------------")
    
    # println("-------------------------------------------------------------")
    # println()
    # println("predictions:") # 6x50 array of Dual{val, deriv w.r.t. loss}.etc type
    # display(pred)
    # println()
    # println("-------------------------------------------------------------")
   

    # if (!is_end_of_epoch_loss)
    #     num_gradient = loss_cos_numerical_gradient(p)
    # end

    # los_cos = new_loss_cos(u0_list[i_exp, :], p, is_end_of_epoch_loss)
    # current_los_cos = loss_cos(p)

    # println("- - - - - - - - - - - - - - - - - - - - - - - - - -")
    # println("current_los_cos: ", current_los_cos)
    # println("- - - - - - - - - - - - - - - - - - - - - - - - - -")
    # loss = mae(ode_data ./ yscale[i_obs], pred ./ yscale[i_obs]) .+ current_los_cos # this is what i need to find a way to change
    # loss is of type Dual{ForwardDiff}, which contains values and derivatives w.r.t. input variables
    loss = mae(ode_data ./ yscale[i_obs], pred ./ yscale[i_obs])
    # loss: Dual{ForwardDiff.Tag{var"#13#14"{Int64},Float64}}
    # (0.1243729180513579,0.0611251492603911,0.7321142565698239,-0.041167254614599615,
    # 0.07142399203396334,0.04181392246023911,0.11011664234222086,0.4429448373235501,
    # 0.1846500040117853,0.03427133752076807)

     if (is_end_of_epoch_loss)
        # println("-------------------------------------------------------------")
        # println()
        # println("loss_cos(p):")
        # display(loss_cos(p))
        # println()
        # println("-------------------------------------------------------------")
        loss = loss + loss_cos(p)
        # loss = loss + loss_collision_limit(p)
        # current_los_cos = loss_cos(p)
        
        # println("--------------------------")
        # println("updating current_los_cos: ")
        # println(current_los_cos)
        # println("--------------------------")
        # loss = loss_cos(p) # only COS loss at the end of the epoch

    end

    # setindex!(loss, loss.value + current_los_cos, 1)
    # new_loss = ForwardDiff.Dual(loss.partials, loss.value + current_los_cos)
    # println("-------------")
    # println("partials: ")
    # display(loss.partials)
    # println("-------------")
    # new_loss = ForwardDiff.Dual{ForwardDiff.Tag{var"#13#14"{Int64},Float64},Float64,10}(loss.value + current_los_cos, loss.partials)
    
    # if (!is_end_of_epoch_loss)
    #   println("-------------------------------------------------------------")
    #   println()
    #   println("loss:")
    #   display(loss)
    #   println()
    #   println("-------------------------------------------------------------")
    # end

    # println("-------------------------------------------------------------")
    # println()
    # println("ode_data ./ yscale[i_obs]:")
    # display(ode_data ./ yscale[i_obs]) # 6x50 array
    # println()
    # println("-------------------------------------------------------------")
   
    # cos_loss = new_loss_compute(current_los_cos, loss.value)
    
    # println("RETURNING")
    # return cos_loss
    # return new_loss
    return loss
end

cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    l_plt = []
    for i in 1:ns
        plt = scatter(tsteps, ode_data[i,:], markercolor=:transparent,
                      title=string(i), label=string("data_", i))
        plot!(plt, tsteps, pred[i,:], label=string("pred_", i))
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false)
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

l_loss_train = []
l_loss_val = []
l_w_out_rxn_1 = Array{Float32}(undef, 1, ns)
l_w_out_rxn_2 = Array{Float32}(undef, 1, ns)
l_w_out_rxn_3 = Array{Float32}(undef, 1, ns)
l_w_b_bias = Array{Float32}(undef, 1, nr)
l_w_in_rxn_1 = Array{Float32}(undef, 1, ns+1)
l_w_in_rxn_2 = Array{Float32}(undef, 1, ns+1)
l_w_in_rxn_3 = Array{Float32}(undef, 1, ns+1)
iter = 1
cb = function (p, loss_train, loss_val, near_the_end)
    global l_loss_train, l_loss_val, iter, l_w_out_rxn_1, species_names, l_w_out_rxn_2, l_w_out_rxn_3, l_w_b_bias, l_w_in_rxn_1, l_w_in_rxn_2, l_w_in_rxn_3
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)

    if iter % n_plot == 0
        display_p(p)
        @printf("min loss train %.4e val %.4e\n", minimum(l_loss_train), minimum(l_loss_val))

        l_exp = randperm(n_exp)[1:1];
        println("update plot for ", l_exp)
        for i_exp in l_exp
            cbi(p, i_exp)
        end

        # plot training/validation progress
        plt_loss = plot(l_loss_train, xscale=:log10, yscale=:log10, 
                        framestyle=:box, label="Training")
        plot!(plt_loss, l_loss_val, label="Validation")
        plot!(xlabel="Epoch", ylabel="Loss", margin=20mm, legend=:outerleft)
      #   png(plt_loss, "figs/loss")
        pdf(plt_loss,"summary_figs/loss")

        # plot coefficients (w_out)
        if near_the_end
            coefficients_rxn_1 = l_w_out_rxn_1[2:end, :] # first row is all 0s -> omit it
            coefficients_rxn_2 = l_w_out_rxn_2[2:end, :]
            coefficients_rxn_3 = l_w_out_rxn_3[2:end, :]
            bias = l_w_b_bias[2:end,:]
            rxn_order_rxn_1 = l_w_in_rxn_1[2:end, :]
            rxn_order_rxn_2 = l_w_in_rxn_2[2:end, :]
            rxn_order_rxn_3 = l_w_in_rxn_3[2:end, :]

            p1 = plot(coefficients_rxn_1[:,1], label=species_names[1])
            for i = 2:size(coefficients_rxn_1, 2)
                plot!(coefficients_rxn_1[:,i], label=species_names[i])
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p1,"summary_figs/coefficients_rxn1")

            p2 = plot(coefficients_rxn_2[:,1], label=species_names[1])
            for i = 2:size(coefficients_rxn_2, 2)
                plot!(coefficients_rxn_2[:,i], label=species_names[i])
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p2,"summary_figs/coefficients_rxn2")

            p3 = plot(coefficients_rxn_3[:,1], label=species_names[1])
            for i = 2:size(coefficients_rxn_3, 2)
                plot!(coefficients_rxn_3[:,i], label=species_names[i])
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p3,"summary_figs/coefficients_rxn3")

            # plot bias (w_b)
            p4 = plot(bias[:,1], label="lnk1")
            for i = 2:size(bias, 2)
                plot!(bias[:,i], label="lnk$i")
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p4,"summary_figs/bias")

            # plot reaction orders (w_in)
            p5 = plot(rxn_order_rxn_1[:,1], label=rxn_order_names[1])
            for i = 2:size(rxn_order_rxn_1, 2) - 1 # -1 because we don't care about the bias (for now)
                plot!(rxn_order_rxn_1[:,i], label=rxn_order_names[i])
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p5,"summary_figs/rxn_order_rxn_1")

            p6 = plot(rxn_order_rxn_2[:,1], label=rxn_order_names[1])
            for i = 2:size(rxn_order_rxn_2, 2) - 1 # -1 because we don't care about the bias (for now)
                plot!(rxn_order_rxn_2[:,i], label=rxn_order_names[i])
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p6,"summary_figs/rxn_order_rxn_2")

            p7 = plot(rxn_order_rxn_3[:,1], label=rxn_order_names[1])
            for i = 2:size(rxn_order_rxn_3, 2) - 1 # -1 because we don't care about the bias (for now)
                plot!(rxn_order_rxn_3[:,i], label=rxn_order_names[i])
            end

            plot!(xlabel="Epoch", ylabel="Value", margin=20mm, legend=:outerleft)
            pdf(p7,"summary_figs/rxn_order_rxn_3")
        end

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    end
    iter += 1;
end


if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    iter += 1;
end

i_exp = 1
epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);
i_epoch = 1
for epoch in epochs
    global p, l_w_out_rxn_1, l_w_out_rxn_2, l_w_out_rxn_3, l_w_b_bias, l_w_in_rxn_1, l_w_in_rxn_2, l_w_in_rxn_3, i_epoch
    for i_exp in randperm(n_exp_train)
        # DISPLAY CODE HERE

        # println("-------------------------------------------------------------")
        # println("p:")
        # display(p) # 25-element array
        # println("-------------------------------------------------------------")
        
        # now, everything is of the right type...
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp, false), p)
        # cos_grad = FiniteDiff.finite_difference_gradient(p -> new_loss_cos(p, true), p) # changed from false
        # cos_grad = loss_cos_numerical_gradient(p)
        # collision_limit_grad = FiniteDiff.finite_difference_gradient(p -> loss_collision_limit(p), p)

        # total_grad = grad .+ (collision_limit_grad .* 0.0001)
        # total_grad = total_grad .+ (cos_grad .* 0.001)

        # println("-------------------------------------------------------------")
        # println("grad:")
        # display(grad) # 25-element array
        # println("-------------------------------------------------------------")
        # println("-------------------------------------------------------------")
        # println("cos_grad:")
        # display(cos_grad) # 25-element array
        # println("-------------------------------------------------------------")

        # original code:
        grad_norm[i_exp] = norm(grad, 2)
        update!(opt, p, grad)

        # new code:
        # grad_norm[i_exp] = norm(total_grad, 2)
        # update!(opt, p, total_grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp, true) # all this does is update the loss_epoch array. it doesn't actually change the model's backprop or anything
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.2e val %.2e gnorm %.1e lr %.1e", 
                                             loss_train, loss_val, mean(grad_norm), opt[1].eta)))
    local w_in, w_b, w_out = p2vec(p)   
    l_w_out_rxn_1 = vcat(l_w_out_rxn_1, transpose(w_out[:,1]))
    l_w_out_rxn_2 = vcat(l_w_out_rxn_2, transpose(w_out[:,2]))
    l_w_out_rxn_3 = vcat(l_w_out_rxn_3, transpose(w_out[:,3]))
    l_w_b_bias = vcat(l_w_b_bias, transpose(w_b))
    l_w_in_rxn_1 = vcat(l_w_in_rxn_1, transpose(w_in[:,1]))
    l_w_in_rxn_2 = vcat(l_w_in_rxn_2, transpose(w_in[:,2]))
    l_w_in_rxn_3 = vcat(l_w_in_rxn_3, transpose(w_in[:,3]))
    near_the_end = i_epoch >= (n_epoch - 10)

    cb(p, loss_train, loss_val, near_the_end);
    i_epoch += 1;
end

# learned weights
print("just finished training...")
w_in, w_b, w_out = p2vec(p)
println("w_in: ")
display(w_in)
println()
println("w_b: ")
display(w_b)
println()
println("w_out: ")
display(w_out)

for i_exp in 1:n_exp
    cbi(p, i_exp)
end