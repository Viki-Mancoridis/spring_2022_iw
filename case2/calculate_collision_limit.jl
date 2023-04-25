T = Float32[332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332; 332] # temperature
kB = 1.380649e-23 # Boltzmann constant (J/K)
mu = Float32[.02791 .02708 .0258] # (g)

sigma = Float32[572.63e-10 511.65e-10 478.9e-10] # collision diameter of Lennard-Jones potential (in m) ==> should be e-9?
epsilon = Float32[6.628e-30 6.259e-30 6.317e-30] # well depth-- was e-23 earlier... ergs --> Joules
# sigma = Float32[819.9e-10 724.8e-10 677.3e-10] # sigma^2 = sigma1^2 + sigma2^2 (m)
# sigma = Float32[6419.9e-10 5624.8e-10 5277.3e-10] # multiply sigma by a factor of --> 6?
# sigma = Float32[1261e-10 724.8e-10 5624.8e-10 5277.3e-10]

Na = 6.022e23 # Avogadro number
arr = Array{Float32}(undef, 30)

# get first estimate for collision limit of rxn 1
T_red_rxn1 = kB * T[1] / epsilon[1]
omega_star = (1.16145 * (T_red_rxn1^-0.14874)) + (0.52487 * exp(-0.7732 * T_red_rxn1)) + (2.16178 * exp(-2.437887 * T_red_rxn1))
k_coll1 = sqrt(8 * pi * kB * T[1] / mu[1]) * (sigma[1]^2) * omega_star * Na

T_red_rxn1 = kB * T[1] / epsilon[2]
omega_star = (1.16145 * (T_red_rxn1^-0.14874)) + (0.52487 * exp(-0.7732 * T_red_rxn1)) + (2.16178 * exp(-2.437887 * T_red_rxn1))
k_coll2 = sqrt(8 * pi * kB * T[1] / mu[2]) * (sigma[2]^2) * omega_star * Na

T_red_rxn1 = kB * T[1] / epsilon[3]
omega_star = (1.16145 * (T_red_rxn1^-0.14874)) + (0.52487 * exp(-0.7732 * T_red_rxn1)) + (2.16178 * exp(-2.437887 * T_red_rxn1))
k_coll3 = sqrt(8 * pi * kB * T[1] / mu[3]) * (sigma[3]^2) * omega_star * Na

println(k_coll1)
println(k_coll2)
println(k_coll3)
