# using Flux: mean
using Statistics: mean

# columns are [C,O,H]
# rows are [TG,ROH,DG,MG,GL,RCOOR]

# this whole analysis should be done on w_out, not w_in
# compositions is modified chemical formulas (to make balanced eqns)
compositions = Float32[9 6 12; 1 1 4; 7 5 11; 5 4 10; 3 3 9; 3 2 5]

w_matrix = ones(7,3)
w_matrix[1,2] = 4
w_matrix[6,1] = 5
# cut off the last row of the matrix
w_matrix = w_matrix[setdiff(1:end, 7), :]
# println("----------")
# println("W MATRIX")
# display(w_matrix)
# println()
# println("----------")
# println("COMPOSITIONS")
# display(compositions)
# println()
# println("----------")


find_elements = function(w_matrix, compositions)
   elements = zeros(3,3)
   carbon_sum = 0
   oxygen_sum = 0
   hydrogen_sum = 0
   for rxn in 1:3
      for species in 1:6
         # global carbon_sum_in, hydrogen_sum_in, oxygen_sum_in, r_sum_in
         carbon_sum += w_matrix[species,rxn] * compositions[species,1]
         oxygen_sum += w_matrix[species,rxn] * compositions[species,2]
         hydrogen_sum += w_matrix[species,rxn] * compositions[species,3]
      end

      elements[rxn,1:end] = [carbon_sum, oxygen_sum, hydrogen_sum]
   end

   return elements
end

# elems = find_elements(w_matrix, compositions)
# println("ELEMENTS")
# display(elems)
# println()
# println("----------")
# println()

test_elements = Float32[1 2 3; 0 1 2; 4 1 3]

n_exp = 50
ns = 6

loss_cos = function(elements)
   squared = elements .^2
   loss = mean(squared) / (n_exp * ns)
   return loss
end

hm = loss_cos(test_elements)
println(hm)
println(hm * 300)

# A = [1 2 3; 4 5 6; 7 8 9]
# B = A .^ 2
# view(A)
# print(typeof(A))
# print(typeof(B))