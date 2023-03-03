using SparseArrays, LinearAlgebra
using NPZ
using JLD
include("mpc_bnb.jl")
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

# Formulation of MPC with state variables
function generate_sparse_MPC_Clarabel(index=2400)
    adaptive_data = npzread("power_converter/results/adaptive_sparseMPC_N=8.npz")
    fixed_data = npzread("power_converter/results/fixed_sparseMPC_N=8.npz")

    x0 = adaptive_data["x0"]
    barA = fixed_data["A"]
    barB = fixed_data["B"]
    barC = fixed_data["C"]
    P0 = fixed_data["P0"]
    q0 = fixed_data["q0"]
    γ = fixed_data["gamma"]
    S = fixed_data["S"]
    R = fixed_data["R"]
    F = fixed_data["F"]   
    horizon = fixed_data["horizon"]
    
    # generate the sparse MPC model
    (nx,nu) = size(barB)   #dimension of x,u in x_{k+1} = ̄Ax_k + ̄Bu_k
    (mF,nF) = size(F)
    (mS,nS) = size(S)

    # generate cost P, q
    Qi = sparse(barC'*barC)
    P = deepcopy(Qi)
    for i = 1:horizon-1
        P = blockdiag(P,γ^i*Qi)
    end
    P = blockdiag(P,sparse(P0))
    P = blockdiag(P, spzeros(nu*horizon,nu*horizon))
    P .*= 2     #later on, we define cost as 0.5*x'Px + q'x
    q = vcat(zeros(nx*horizon), 2*q0*γ^horizon, zeros(nu*horizon))

    # equality constraints Gx = h
    G1 = blockdiag(spdiagm(ones(nx)), -spdiagm(ones(nx*horizon)))
    G2 = kron(spdiagm(ones(horizon)),barA)
    G3 = kron(spdiagm(ones(horizon)),barB)
    G = [G1 + [spzeros(nx,nx*(horizon+1));G2 spzeros(nx*horizon,nx)] vcat(spzeros(nx,nu*horizon),G3)]
    h = vcat(x0[:,index],zeros((nx+nu)*horizon,1))

    #inequality constraints Ax ≤ b
    A = sparse([-S R; zeros(mF,nS) F])
    b = vcat(zeros(mS,1), ones(mF,1))

    # box constraints lb ≤ Ib*x ≤ ub
    Ib = sparse([zeros(nu*horizon,nx*(horizon+1)) 1.0*I])
    lb = -ones(nu*horizon);
    ub = ones(nu*horizon);

    # construct augmented ̃A and ̃b containing all constraints
    Ã = vcat(G, A, -Ib, Ib)  # ATTENTION: to be consistent with toy problem, have ordered 1. lb 2.ub
    b̃ = vcat(h, b, -lb, ub)
    cones = [Clarabel.ZeroConeT(length(h)), Clarabel.NonnegativeConeT(length(b) + 2*length(lb))]
    return P, q, Ã, b̃, cones, lb, ub
end

# Formulation of MPC without state variables
function generate_dense_MPC_Clarabel(index=2400)
    adaptive_data = npzread("power_converter/results/adaptive_denseMPC_N=8.npz")
    fixed_data = npzread("power_converter/results/fixed_denseMPC_N=8.npz")
    P = fixed_data["P"]
    q = adaptive_data["q"][:,index]
    A = fixed_data["A"] 
    b = zeros(size(A,1))
    index_set = fixed_data["i_idx"] .+ 1 # offset by 1 since extracted from python array starting from 0 not 1 as in julia

    # construct augmented A and b containing all inequality constraints
    l = b + fixed_data["l"] #extended array of lower bounds
    u = b + adaptive_data["u"][:,index]
    lb = fixed_data["i_l"] #lower bound on integer variables
    ub = fixed_data["i_u"] # upper bound on integer variables
    dim = length(lb)
    # for -Ax ≤ -l constraints, consider only last 3N ([dim+1:end]) rows since no lower bound for (R-SB)*U ≤ S*X constraints (set to Inf)
    Ã = vcat(A, -A[dim+1:end,:], -I, I)  # ATTENTION: to be consistent with toy problem, have ordered 1. lb 2.ub
    b̃ = vcat(u, -l[dim+1:end], -lb, ub)
    cones = [Clarabel.NonnegativeConeT(length(b̃))]
    return sparse(P), q, sparse(Ã), b̃, cones, index_set, sparse(A), b, l, u, lb, ub
end

function generate_MPC_Clarabel(index=2400)
    adaptive_data = npzread("paper_test_miqp-main\\mpc_data\\N=8\\adaptive_data.npy")
    fixed_data = npzread("paper_test_miqp-main\\mpc_data\\N=8\\matrice_data.npy")
    P = fixed_data["P"]
    q = adaptive_data["q_array"][index,:]
    A = fixed_data["A"] 
    b = zeros(size(A,1))
    index_set = fixed_data["i_idx"] .+ 1 # offset by 1 since extracted from python array starting from 0 not 1 as in julia

    println("Conditioning number of P: ",cond(P))

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

function factorize_optimization_based_matrix(data,I_B,G, σ, η, γ)
    eyemat=Matrix(1.0I, data.n, data.n) 
    ldltS = ldlt([Symmetric(data.P+σ*eyemat)     I_B    G';
            I_B   -η*eyemat         zeros(data.n,data.n);
            G     zeros(data.n,data.n)     -γ*eyemat])
    return ldltS
end
    
without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]
percentage_iter_reduction = Float64[]
start_horizon = 2200
end_horizon = 2300
for i = start_horizon:end_horizon
    printstyled("Horizon iteration: ", i, "\n", color = :magenta)
    P, q, Ã, b̃, s, i_idx,A, b, l, u, lb, ub= generate_MPC_Clarabel(i)
    n = length(q)

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
    #= println("P: ", P)
    println("q : ", q)
    println("A : ", A)
    println("b : ", b)
    println("s : ", s)  =#
    λ=0.99
    η= 1e-3 # set to 1000.0 to disable optimise_correction entirely
    γ = 1e-3
    ϵ = 1e-8

    settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ã, b̃,s, settings)
    ldltS = factorize_optimization_based_matrix(Clarabel.data,I_B,G,σ,η, γ)
    
    base_solution = Clarabel.solve!(solver)
    println("Clarabel base result " ,base_solution, " with base_solution ", base_solution.x)

#start bnb loop
    println("STARTING CLARABEL BNB LOOP ")

    best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(solver, base_solution,n,ϵ, i_idx, true, true, false, λ,η,true,false,ldltS) 

    
    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Number of early terminated nodes: ", early_num)
    println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
    println("Gurobi base_solution: ", objective_value(model) , " using ", value.(model[:x])) 
    println(" ")
    diff_sol_vector= feasible_solution - value.(model[:x])
    diff_solution=round(norm(diff_sol_vector),digits=5)
    diff_obj = round(best_ub-objective_value(model),digits=6)
    if ~iszero(diff_solution) || ~iszero(diff_obj)
        println("Solution diff: ",diff_solution, "Obj difference: ", diff_obj)
        println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
        println("Horizon iteration number: ", i)
        error("Solutions differ!")
    end

    # count QP iterations    
    printstyled("Total net iter num (with early_term_enable): ", total_iter-fea_iter, "\n", color = :green)
    solver_without   = Clarabel.Solver()

    Clarabel.setup!(solver_without, P, q, Ã, b̃,s, settings)
    
    base_solution_without = Clarabel.solve!(solver_without)
    best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(solver_without, base_solution_without,n,ϵ, i_idx, true, false, false,λ,η,false,false,ldltS) 
    println("Found objective without early_term: ", best_ub_without)
    println("Number of early terminated nodes (without): ", early_num_without)
    printstyled("Total net iter num (without): ", total_iter_without - fea_iter_without, "\n", color = :green)
    reduction = 1 - (total_iter - fea_iter)/(total_iter_without - fea_iter_without)
    println("Reduced iterations (percentage): ", reduction)
    append!(without_iter_num, total_iter_without)
    append!(with_iter_num, total_iter)
    append!(percentage_iter_reduction, reduction)
    if (fea_iter == fea_iter_without)
        append!(first_iter_num, fea_iter)
    end

    println(" ")

    
end 
   
save("mimpc_iterations_N=8.jld", "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction)
