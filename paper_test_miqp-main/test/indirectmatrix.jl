using IterativeSolvers

A = [1.0 0; 0 1]
b = [2.0;3]
c = IterativeSolvers.cg!(zero(b), A, b)
println(c)
