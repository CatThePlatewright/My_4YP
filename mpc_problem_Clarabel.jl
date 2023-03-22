using SparseArrays, LinearAlgebra, QDLDL
using NPZ
using JLD
using Printf
include("mpc_bnb.jl")
using Gurobi

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
N=8
# Formulation of MPC with state variables
function generate_sparse_MPC_Clarabel(index=2400)
    adaptive_data = npzread(@sprintf("power_converter/results/adaptive_sparseMPC_N=%d.npz",N)) # we have N = 2,4,6,8,10,12
    fixed_data = npzread(@sprintf("power_converter/results/fixed_sparseMPC_N=%d.npz",N))
    x0 = adaptive_data["x0"]
    barA = fixed_data["A"]
    barB = fixed_data["B"]
    barC = fixed_data["C"]
    P0 = fixed_data["P0"]
    q0 = fixed_data["q0"] # has length of variable x at horizon N, so nx entries (12 for N=2)
    γ = fixed_data["gamma"]
    S = fixed_data["S"]
    R = fixed_data["R"]
    F = fixed_data["F"]   
    horizon = fixed_data["horizon"]
    println("horizon: ", horizon)
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
    q = vcat(zeros(nx*horizon), 2*q0*γ^horizon, zeros(nu*horizon))[:]

    # equality constraints Gx = h
    G1 = blockdiag(spdiagm(ones(nx)), -spdiagm(ones(nx*horizon)))
    G2 = kron(spdiagm(ones(horizon)),barA)
    G3 = kron(spdiagm(ones(horizon)),barB)
    G = [G1 + [spzeros(nx,nx*(horizon+1));G2 spzeros(nx*horizon,nx)] vcat(spzeros(nx,nu*horizon),G3)]
    h = vcat(x0[:,index],zeros(nx*horizon,1))[:]

    #inequality constraints Ax ≤ b
    A = sparse([-S R; zeros(mF,nS) F])
    b = vcat(zeros(mS,1), ones(mF,1))

    # box constraints lb ≤ Ib*x ≤ ub
    Ib = sparse([zeros(nu*horizon,nx*(horizon+1)) 1.0*I])
    lb = -ones(nu*horizon);
    ub = ones(nu*horizon);

    # construct augmented ̃A and ̃b containing all constraints
    Ã = vcat(G, A, -Ib, Ib)  # ATTENTION: to be consistent with toy problem, have ordered 1. lb 2.ub
    b̃ = vcat(h, b, -lb, ub)[:]
    i_idx = collect(lastindex(q)-horizon*nu+1:lastindex(q)) #6 is the dimension of u for each time horizon
    cones = [Clarabel.ZeroConeT(length(h)), Clarabel.NonnegativeConeT(length(b) + 2*length(lb))]
    return sparse(P), q, G,h, Ib, sparse(A), b, sparse(Ã), b̃, cones, lb, ub, i_idx
end
function factorize_optimization_based_matrix(data,Ib,G, σ, η, γ)
    n = data.n 
    g_width = size(G)[1]
    ib_width = size(Ib)[1]
    S = [Symmetric(data.P+σ*Matrix(1.0I,n,n))     Ib'    G';
    Ib   -η*Matrix(1.0I, ib_width, ib_width)        zeros(ib_width,g_width);
    G     zeros(g_width, ib_width)     -γ*Matrix(1.0I, g_width,g_width)]
    ldltS = ldlt(sparse(S))
    return ldltS
end
# Formulation of MPC without state variables
function generate_dense_MPC_Clarabel(index=2400)
    adaptive_data = npzread("power_converter/results/adaptive_denseMPC_N=2.npz")
    fixed_data = npzread("power_converter/results/fixed_denseMPC_N=2.npz")
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
    
without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]
percentage_iter_reduction = Float64[]
start_horizon = 2200
end_horizon = 2300
num_errors = 0
for i = start_horizon:end_horizon
#=    printstyled("Horizon iteration: ", i, "\n", color = :magenta)
    P, q, G,h, Ib, A, b, Ã, b̃, cones, lb, ub, i_idx= generate_sparse_MPC_Clarabel(i)
    n = length(q)
    Nnu = length(i_idx)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    @variable(model, x[1:n])
    set_integer.(x[i_idx])  #set integer constraints
    @objective(model, Min, 0.5*x'*P*x + q' * x )
    dim = length(lb)
    @constraints(model, begin
        G*x .== h
        A*x .<= b
        # box constraints lb ≤ Ib*x ≤ ub
        Ib*x .>= lb
        Ib*x .<= ub
    end)
    optimize!(model)
    uopt1 = value.(x)[end - Nnu + 1:end]
    println("Gurobi base_solution: ", objective_value(model) , " using ", uopt1) 
    
    λ=0.99
    η= 1000 # penalisation for y_B
    γ = 100 # penalisation for z
    σ = 1e-7 # perturbation added to P in optimization matrix

    ϵ = 1e-6
 
    settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ã, b̃,cones, settings)
    ldltS = factorize_optimization_based_matrix(solver.data,Ib,G,σ,η, γ)
    base_solution = Clarabel.solve!(solver)
    println("Clarabel base result " ,base_solution, " with base_solution ", base_solution.x[end - Nnu + 1:end])

    #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")

    best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(i, solver, base_solution,n,ϵ, i_idx, true, true, false, λ,ldltS,true,false) 

    
    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Number of early terminated nodes: ", early_num)
    println("Found objective: ", best_ub, " using ", round.(feasible_solution[end - Nnu + 1:end],digits=3))
    println("Gurobi base_solution: ", objective_value(model) , " using ", uopt1) 
    println(" ")
    diff_sol_vector= feasible_solution[end - Nnu + 1:end] - uopt1
    diff_solution=round(norm(diff_sol_vector),digits=3)
    diff_obj = round(best_ub-objective_value(model),digits=6)
    if ~iszero(diff_solution) || ~iszero(diff_obj)
        println("Solution diff: ",diff_solution, "Obj difference: ", diff_obj)
        println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
        println("Horizon iteration number: ", i)
        error("Solutions differ!")
        #num_errors = num_errors +1
    end
    

    # count QP iterations    
    printstyled("Total net iter num (with early_term_enable): ", total_iter-fea_iter, "\n", color = :green)
    solver_without   = Clarabel.Solver()

    Clarabel.setup!(solver_without, P, q, Ã, b̃,cones, settings)
    
    base_solution_without = Clarabel.solve!(solver_without)
    best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(i,solver_without, base_solution_without,n,ϵ, i_idx, true, false, false,λ,ldltS,false,false) 
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
    println("Total iterations: ", total_iter, " ", total_iter_without)
    println(" ") 
end=#

# dense formulation for MIMPC:
    printstyled("Horizon iteration: ", i, "\n", color = :magenta)
    P, q, Ã, b̃, s, i_idx,A, b, l, u, lb, ub= generate_dense_MPC_Clarabel(i)
    n = length(q)

    model2 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model2, "OutputFlag", 0)
    @variable(model2, x[1:n])
    set_integer.(x[i_idx])  #set integer constraints
    @objective(model2, Min, 0.5*x'*P*x + q' * x )
    dim = length(lb)
    @constraints(model2, begin
        x .>= lb
        x .<= ub
    # for -Ax ≤ -l constraints, consider only last 3N rows since no lower bound for (R-SB)*U ≤ S*X constraints
        b[dim+1:end] + A[dim+1:end,:]*x .>= l[dim+1:end] # TOASK: confirm that A should be +fixed_data not -?
        b + A*x .<= u
    end)
    optimize!(model2)
    uopt2 = value.(x)
    println("Gurobi base_solution: ", objective_value(model2) , " using ", uopt2) 
    #check_flag = all(uopt1 .== uopt2)
    #@assert(check_flag == true)
    #printstyled("Same solution:", check_flag, "\n",color=:green)
    λ=0.99
    ϵ = 1e-6
    ldltS = nothing

    settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ã, b̃,s, settings)
    #ldltS = factorize_optimization_based_matrix(Clarabel.solver.data,I_B,G,σ,η, γ)
    base_solution = Clarabel.solve!(solver)
    println("Clarabel base result " ,base_solution, " with base_solution ", base_solution.x)

    #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")

    best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(i,solver, base_solution,n,ϵ, i_idx, true, true, true, λ,ldltS,true,false) 

    
    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Number of early terminated nodes: ", early_num)
    println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
    println("Gurobi base_solution: ", objective_value(model2) , " using ", uopt2) 
    println(" ")
    diff_sol_vector= feasible_solution - value.(model2[:x])
    diff_solution=round(norm(diff_sol_vector),digits=5)
    diff_obj = round(best_ub-objective_value(model2),digits=6)
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
    best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(i,solver_without, base_solution_without,n,ϵ, i_idx, true, false, true,λ,ldltS,false,false) 
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
save((@sprintf("mimpc_iterations_N=%d_warmstart_new.jld",N)), "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction)
   
