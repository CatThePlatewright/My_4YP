using LinearAlgebra, SparseArrays, Random, Test
# using Plots
using Debugger
using Revise
# this shows how to load the maros meszaros problem data from the .jld2 file
using FileIO, JLD, JLD2 # to load the matrices from the file
using LinearAlgebra, SparseArrays
using PyPlot
using JuMP, Gurobi

includet("src\\admm_operator.jl")
includet("src\\branch_and_bound.jl")
includet("src\\cuts.jl")
includet("src\\generate_sample.jl")

# function miqp()
#generate data
# cond_num = Inf
m = 200
n = 50
inum = 5
sparsity = 0.7
P_rank = Int(floor(n/2))
# P_rank = Int(floor(n))
P, q, A, b, l, u, lb, ub, i_idx = generate_random_sample(m, n, inum, sparsity, P_rank)
cond_num = cond(Matrix(P))
println("condition number of P is ", cond_num)


P = load("original-problem.jld","P")
q = load("original-problem.jld","q")
A = load("original-problem.jld","A")
b = load("original-problem.jld","b")
l = load("original-problem.jld","l")
u = load("original-problem.jld","u")
lb = load("original-problem.jld","lb")
ub = load("original-problem.jld","ub")
i_idx = load("original-problem.jld","i_idx")
m = length(b)
n = length(lb)
A = sparse(A)

# P = load("problem.jld","P")
# q = load("problem.jld","q")
# A = load("problem.jld","A")
# b = load("problem.jld","b")
# l = load("problem.jld","l")
# u = load("problem.jld","u")
# lb = load("problem.jld","lb")
# ub = load("problem.jld","ub")
# i_idx = load("problem.jld","i_idx")
# m = length(b)
# n = length(lb)
# A = sparse(A)

# model = Model(Gurobi.Optimizer)
# @variable(model, x[1:n]);
# set_integer.(x[i_idx])  #set integer constraints
# @objective(model, Min, 0.5*x'*P*x + q' * x )
# dim = length(lb)
# @constraints(model, begin
# 	x .>= lb
# 	x .<= ub
# 	b - A*x .>= l
# 	b - A*x .<= u
# end)
# optimize!(model)
# println("Gurobi solution: ", JuMP.objective_value(model))
save("original-problem.jld", "P", Matrix(P), "q", q, "A", Matrix(A), "b", b, "l", l, "u", u, "lb", lb, "ub", ub, "i_idx", i_idx)


# set up operator
sigma = 1e-6;

#set penalty parameter
ρ = 1e0 * ones(m);
eq_ind = broadcast(==, l, u)
@. ρ[eq_ind] = 1e3*ρ[eq_ind]
ρ_x = 1e0 * ones(n);
eq_ind_x = broadcast(==, lb, ub)
@. ρ_x[eq_ind_x] = 1e3*ρ_x[eq_ind_x]

α = 1.6;
max_iter = 100000;
eps_abs = 1e-4;
eps_rel = 1e-4; #infeasibility detection
eps_int = 1e-3; #tolerance for integer check
constraints = [:qp, l, u]
settings = Dict([("eps_int",eps_int)])
s_early = true
# s_early = false

# operator = MIADMMOperator(P, q, A, b, constraints, lb, ub, sigma, ρ, ρ_x, α, max_iter, eps_abs, eps_rel);
LR_operator = OuterMIQP(deepcopy(settings), deepcopy(P), deepcopy(q), deepcopy(A), deepcopy(b), deepcopy(constraints), deepcopy(lb), deepcopy(ub), deepcopy(i_idx), deepcopy(sigma), deepcopy(ρ), deepcopy(ρ_x), deepcopy(α), deepcopy(max_iter), deepcopy(eps_abs), deepcopy(eps_rel), s_early);

# initialise ADMM operator variable v. v[1:n] corresponds to x; v[n+1; 2n] corresonds to x box constraints; v[2n+1; 2n+m] corresponds to equality constraints

v = zeros(n + m + n);
while !isempty(LR_operator.leaves)
	cur_leaf = select_leaf(LR_operator)
	# println("Node ", LR_operator.iter_num)

	#solve the relaxed QP problem at leaf
	# start_time = time();

	solve_relaxed!(v, cur_leaf, LR_operator)
	# end_time = time()
	# println("Current obj_val: ", cur_leaf.lower)

	#branch and bounds
	branch_and_bound(cur_leaf, LR_operator)

	# println("Feasible value: ", LR_operator.upper)
	# println("Lower bound: ", LR_operator.lower)

	#check termination
	if isempty(LR_operator.leaves)
		println("All nodes are solved, terminate with value ", cur_leaf.operator.sm.cinv*LR_operator.upper)
		break
	# elseif (LR_operator.upper - LR_operator.lower) < 1e-3*abs(LR_operator.upper)
	# 	println("The optimality gap is small enough, terminate with value ", LR_operator.upper)
	# 	break
	elseif LR_operator.iter_num > 10000
		println("Exceed maximum number of iterations", cur_leaf.operator.sm.cinv*LR_operator.upper)
		break
	end

end
println("with #:", LR_operator.iter_num)

# end
