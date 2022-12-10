using LinearAlgebra, SparseArrays, Random, Test
# this shows how to load the maros meszaros problem data from the .jld2 file
using FileIO, JLD, JLD2 # to load the matrices from the file
using JuMP, Gurobi, OSQP, COSMO

P = load("problem.jld","P")
q = load("problem.jld","q")
A = load("problem.jld","A")
b = load("problem.jld","b")
l = load("problem.jld","l")
u = load("problem.jld","u")
lb = load("problem.jld","lb")
ub = load("problem.jld","ub")
i_idx = load("problem.jld","i_idx")
m = length(b)
n = length(lb)
A = sparse(A)

model = Model(with_optimizer(COSMO.Optimizer, max_iter = 100000, adaptive_rho = true))
@variable(model, x[1:n]);
# set_integer.(x[i_idx])  #set integer constraints
@objective(model, Min, 0.5*x'*P*x + q' * x )
dim = length(lb)
@constraints(model, begin
	x .>= lb
	x .<= ub
	b - A*x .>= l
	b - A*x .<= u
end)
optimize!(model)
# println("OSQP solution: ", JuMP.objective_value(model))
