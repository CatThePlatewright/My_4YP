using SparseArrays, LinearAlgebra
using NPZ
include("direct_Clarabel_large_augmented.jl")
"""
ADMM problem format:
min 0.5 x'Px + q'x
s.t.  l ≤ Ax ≤u,
	  x ∈ Z, relaxed to [lx, ux] (initially to [-1,1])

convert into IPM format:
min ...
s.t. Ãx ≤ b̃ 
where Ã = [A -A -I I]' and b̃ = [u -l -lx ux]'
"""

function generate_MPC_Clarabel(index=2400)
    adaptive_data = npzread("paper_test_miqp-main\\mpc_data\\N=2\\adaptive_data.npy")
    fixed_data = npzread("paper_test_miqp-main\\mpc_data\\N=2\\matrice_data.npy")
    P = fixed_data["P"]
    q = adaptive_data["q_array"][index,:]
    A = fixed_data["A"] # TOASK: removed minus sign for ADMM? do we still need negative sign with Clarabel uses Ax+s=b?
    b = zeros(size(A,1))
    index_set = fixed_data["i_idx"] .+ 1 # offset by 1 since extracted from python array starting from 0 not 1 as in julia

    println(cond(P))

    # construct augmented A and b containing all inequality constraints
    l = b + fixed_data["l"] #extended array of lower bounds
    u = b + adaptive_data["q_u_array"][index,:]
    lb = fixed_data["i_l"] #lower bound on integer variables
    ub = fixed_data["i_u"] # upper bound on integer variables 
    dim = length(lb)
    # for -Ax ≤ -l constraints, consider only last 3N ([dim+1:end]) rows since no lower bound for (R-SB)*U ≤ S*X constraints (set to Inf)
    Ã = vcat(A, -A[dim+1:end,:], -I, I)  # ATTENTION: to be consistent with toy problem, have ordered 1. lb 2.ub
    b̃ = vcat(u, -l[dim+1:end], -lb, ub)
    s = [Clarabel.NonnegativeConeT(length(b̃))]
    return sparse(P), q, sparse(Ã), b̃, s, index_set, sparse(A), b, l, u, lb, ub
end
function main_mpc()
    i = 2400
    println("Iteration: ", i)
	P, q, Ã, b̃, s, i_idx,A, b, l, u, lb, ub= generate_MPC_Clarabel(i)
	n = length(q)
    #= println("P: ", P)
    println("q : ", q)
    println("A : ", A)
    println("b : ", b)
    println("s : ", s)  =#
    ϵ = 0.00000001

    settings = Clarabel.Settings(verbose = true, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ã, b̃,s, settings)
    
    base_solution = Clarabel.solve!(solver)
    println(base_solution)
    println("Clarabel base result " ,base_solution, " with base_solution ", base_solution.x)

  #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")

    time_taken = @elapsed begin
     best_ub, feasible_base_solution = branch_and_bound_solve(solver, base_solution,n,ϵ, i_idx) 
    end
    println("Time taken by bnb loop: ", time_taken)
    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Found objective: ", best_ub, " using ", round.(feasible_base_solution,digits=3))

    println(" ")

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
	@variable(model, x[1:n])
	set_integer.(x[i_idx])  #set integer constraints
	@objective(model, Min, 0.5*x'*P*x + q' * x )
	dim = length(lb)
	@constraints(model, begin
	    x .>= lb
	    x .<= ub
     # for -Ax ≤ -l constraints, consider only last 3N rows since no lower bound for (R-SB)*U ≤ S*X constraints
	    b[dim+1:end] + A[dim+1:end,:]*x .>= l[dim+1:end] # TOASK: confirm that A should be +fixed_data not -?
	    b + A*x .<= u
	end)

	optimize!(model)
    println("Gurobi base_solution: ", objective_value(model) , " using ", value.(model[:x])) 
    println("Compare with exact: ", round(norm(feasible_base_solution - value.(model[:x])),digits=5), " ",round(best_ub-objective_value(model),digits=6))
    
    return 
end
main_mpc()