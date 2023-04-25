x = 8.337
difference = abs(x - round(x))
println(difference)

w_matrix = ones(7,3)
w_matrix[1,2] = 1.58
w_matrix[6,1] = 5.24

display(w_matrix)

differences = [abs(x - round(x)) for x in w_matrix]

display(differences)