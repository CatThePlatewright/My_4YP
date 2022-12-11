using COSMO, LinearAlgebra, JuMP, Random
using SparseArrays
using Revise
using FileIO, JLD, JLD2
using JuMP, Gurobi

# # Define problem matrices
# q = [1; 1.; 2.];
# P = [2. -1. -1.; -1. 3. 1.; -1. 1. 2.];
# A = -[2. 1. 1.];
# l = [-1.];
# u = [3.];
# m, n = size(A);
# li = -100*ones(n);	#relaxed box constraints for integer constraints
# ui = 100*ones(n);		#use datatype Float64 for simplicity
# b = zeros(m);
# li[1] = -0.0
#
# Solve with JuMP + COSMO
P = load("data\\original-problem.jld","P")
q = load("data\\original-problem.jld","q")
A = load("data\\original-problem.jld","A")
b = load("data\\original-problem.jld","b")
l = load("data\\original-problem.jld","l")
u = load("data\\original-problem.jld","u")
lb = load("data\\original-problem.jld","lb")
ub = load("data\\original-problem.jld","ub")
i_idx = load("data\\original-problem.jld","i_idx")

# P = load("problem.jld","P")
# q = load("problem.jld","q")
# A = load("problem.jld","A")
# b = load("problem.jld","b")
# l = load("problem.jld","l")
# u = load("problem.jld","u")
lb = load("data\\problem.jld","lb")
ub = load("data\\problem.jld","ub")
i_idx = load("data\\problem.jld","i_idx")
m = length(b)
n = length(lb)
A = sparse(A)

# includet("src\\generate_sample.jl")
# P, q, A, b, l, u, lb, ub, i_idx = generate_MPC(100)
# m,n = size(A)

#Gurobi formulation
model = Model(Gurobi.Optimizer)
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
# println("Gurobi solution: ", JuMP.objective_value(model))

# COSMO
model = JuMP.Model(with_optimizer(COSMO.Optimizer, merge_strategy = COSMO.NoMerge, max_iter = 10000, complete_dual = true));
@variable(model, x[1:n]);
@objective(model, Min, 0.5*x'*P*x + q' * x )
# @constraint(model, Symmetric(B - A1  .* x[1] - A2 .* x[2] )  in JuMP.PSDCone());
dim = length(lb)
@constraints(model, begin
    x .>= lb
    x .<= ub
    b - A*x .>= l
    # b[dim+1:end] - A[dim+1:end,:]*x .>= l[dim+1:end]
    b - A*x .<= u
end)
JuMP.optimize!(model)

#Reformulation
# model = COSMO.Model();
# settings = COSMO.Settings(rho = 1.0, scaling = 0, adaptive_rho = false, decompose = false, compact_transformation = false);
# settings = COSMO.Settings();
# boxconstraint = COSMO.Constraint(-A, b, COSMO.Box(l,u))
# idenconstraint = COSMO.Constraint(Matrix(1.0*I,n,n), zeros(n), COSMO.Box(lb,ub))
# assemble!(model, P, q, [boxconstraint; idenconstraint], settings = settings);
# result = COSMO.optimize!(model)
