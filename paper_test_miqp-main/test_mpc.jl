using LinearAlgebra, SparseArrays, Random, Test
# using Plots
using Debugger
using Revise
# this shows how to load the maros meszaros problem data from the .jld2 file
using FileIO, JLD, JLD2 # to load the matrices from the file
using JuMP, Gurobi

using TimerOutputs

includet("src\\admm_operator.jl")
includet("src\\branch_and_bound.jl")
# includet("src\\cuts.jl")
includet("src\\generate_sample.jl")

start_horizon = 2400
end_horizon = 2400
# 104, 108, 109, 1182

without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]

for i = start_horizon:end_horizon
	println("Iteration: ", i)
	P, q, A, b, l, u, lb, ub, i_idx = generate_MPC(i)
	m, n = size(A)

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
	max_iter = 20000;
	eps_abs = 1e-5;
	eps_rel = 1e-4; #infeasibility detection
	eps_int = 1e-3; #tolerance for integer check
	constraints = [:qp, l, u]
	settings = Dict([("eps_int",eps_int)])


	f_operator = OuterMIQP(deepcopy(settings), deepcopy(P), deepcopy(q), deepcopy(A), deepcopy(b), deepcopy(constraints), deepcopy(lb), deepcopy(ub), deepcopy(i_idx), deepcopy(sigma), deepcopy(ρ), deepcopy(ρ_x), deepcopy(α), deepcopy(max_iter), deepcopy(eps_abs), deepcopy(eps_rel), false);
	t_operator = OuterMIQP(deepcopy(settings), deepcopy(P), deepcopy(q), deepcopy(A), deepcopy(b), deepcopy(constraints), deepcopy(lb), deepcopy(ub), deepcopy(i_idx), deepcopy(sigma), deepcopy(ρ), deepcopy(ρ_x), deepcopy(α), deepcopy(max_iter), deepcopy(eps_abs), deepcopy(eps_rel), true);

	#with early termination
	# initialise ADMM operator variable v
	v = zeros(n + m + n);
	while !isempty(t_operator.leaves)
		cur_leaf = select_leaf(t_operator)
		# println("Node ", t_operator.iter_num, " with lb ", cur_leaf.operator.lb[t_operator.i_idx], "  u ", cur_leaf.operator.ub[t_operator.i_idx])

		#solve the relaxed QP problem at leaf
		solve_relaxed!(v, cur_leaf, t_operator)

		#branch and bounds
		branch_and_bound(cur_leaf, t_operator)

		# println("leaves: ", length(t_operator.leaves))

		#check termination
		if isempty(t_operator.leaves)
			println("All nodes are solved, terminate with value ", cur_leaf.operator.sm.cinv*t_operator.upper)
			break
		elseif t_operator.iter_num > 10000
			println("Exceed maximum number of iterations", cur_leaf.operator.sm.cinv*t_operator.upper)
			break
		end
	end
	println("with #:", t_operator.iter_num)

	#without early termination
	# initialise ADMM operator variable v
	v = zeros(n + m + n);
	while !isempty(f_operator.leaves)
		cur_leaf = select_leaf(f_operator)
		# println(cond([Matrix(cur_leaf.operator.P) Matrix(cur_leaf.operator.A'); Matrix(cur_leaf.operator.A) zeros(m,m)]))
		# println("Node ", f_operator.iter_num, " with lb ", cur_leaf.operator.lb[f_operator.i_idx], "  u ", cur_leaf.operator.ub[f_operator.i_idx])

		#solve the relaxed QP problem at leaf
		# start_time = time();
		solve_relaxed!(v, cur_leaf, f_operator)
		# end_time = time()
		# println("Iteration requirement: ", cur_leaf.operator.iter)
		# println("Iteration status: ", cur_leaf.operator.status)

		#branch and bounds
		branch_and_bound(cur_leaf, f_operator)

		# println("Feasible value: ", f_operator.upper)
		# println("Lower bound: ", f_operator.lower)
		# println("leaves: ", length(f_operator.leaves))

		#check termination
		if isempty(f_operator.leaves)
			println("All nodes are solved, terminate with value ", cur_leaf.operator.sm.cinv*f_operator.upper)
			break
		# elseif (f_operator.upper - f_operator.lower) < 1e-3*abs(f_operator.upper)
		# 	println("The optimality gap is small enough, terminate with value ", f_operator.upper)
		# 	break
		elseif f_operator.iter_num > 10000
			println("Exceed maximum number of iterations", cur_leaf.operator.sm.cinv*f_operator.upper)
			break
		end
	end
	println("without #: ", f_operator.iter_num)


	# count QP iterations
	println("Total iter num (with): ", t_operator.total_iter - t_operator.fea_iter, "       Total iter num (without): ", f_operator.total_iter - f_operator.fea_iter)
	println("Reduced iterations (percentage): ", 1 - (t_operator.total_iter - t_operator.fea_iter)/(f_operator.total_iter - f_operator.fea_iter))
	append!(without_iter_num, f_operator.total_iter)
	append!(with_iter_num, t_operator.total_iter)
	if (t_operator.fea_iter == f_operator.fea_iter)
		append!(first_iter_num, t_operator.fea_iter)
	end

	# println(ub)
	# cur_leaf = select_leaf(f_operator)
	# op = cur_leaf.operator
	# P = op.P
	# q = op.q
	# A = op.A
	# b = op.b
	# l = op.constraints[2]
	# u = op.constraints[3]
	# lb = op.sm.Dinv * lb
	# ub = op.sm.Dinv * ub
	model = Model(Gurobi.Optimizer)
	@variable(model, x[1:n]);
	set_integer.(x[i_idx])  #set integer constraints
	@objective(model, Min, 0.5*x'*P*x + q' * x )
	dim = length(lb)
	@constraints(model, begin
	    x .>= lb
	    x .<= ub
	    b[dim+1:end] - A[dim+1:end,:]*x .>= l[dim+1:end]
	    b - A*x .<= u
	end)

	optimize!(model)
	println("Gurobi solution: ", JuMP.objective_value(model))
end
# end

#NOTE: this was not commented out (Cat)
#save("data\\mimpc_comp.jld", "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num)

# save("data\\problem.jld", "P", Matrix(P), "q", q, "A", Matrix(A), "b", b, "l", l, "u", u, "lb", lb, "ub", ub, "i_idx", i_idx)
