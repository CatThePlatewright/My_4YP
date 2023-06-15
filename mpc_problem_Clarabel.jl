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
function generate_sparse_MPC_Clarabel(index=2400, sample=0)
    adaptive_data = npzread(@sprintf("power_converter/results/adaptive_sparseMPC_N=%d_%d_.npz",N,sample)) # we have N = 2,4,6,8,10,12
    fixed_data = npzread(@sprintf("power_converter/results/fixed_sparseMPC_N=%d_%d_.npz",N,sample))
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
    horizon = N
    # println("horizon: ", horizon)
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
    luS = lu(sparse(S))
    return luS
end
# Formulation of MPC without state variables
function generate_dense_MPC_Clarabel(index=2400)
    adaptive_data = npzread(@sprintf("power_converter/results/adaptive_denseMPC_N=%d.npz",N))
    fixed_data = npzread(@sprintf("power_converter/results/fixed_denseMPC_N=%d.npz",N))
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

#function save time 
function save_time(i, t, start_horizon, first_time, with_time, fea_time, total_time)
    first_time[i - start_horizon + 1,t] = fea_time
    with_time[i - start_horizon + 1,t] = total_time
    @assert(total_time >= fea_time)
end

start_horizon = 2200
end_horizon = 2300      #2300
num_time = end_horizon - start_horizon + 1
num_errors = 0
repeat_time = 5
num_sample = 1

warm_start = true       #whether warm-start
debug_print = false
sparsity = "sparse"     #choose whether we test on 'dense' or 'sparse' problems


# Optimality data
opt_cost = Vector{Float64}(undef,num_time)
if sparsity == "sparse"
    P, q, G,h, Ib, A, b, Ã, b̃, cones, lb, ub, i_idx= generate_sparse_MPC_Clarabel(1,0)
    Nnu = length(i_idx)
    opt_x = Matrix{Float64}(undef,Nnu,num_time)
else
    P, q, Ã, b̃, s, i_idx,A, b, l, u, lb, ub= generate_dense_MPC_Clarabel(1)
    opt_x = Matrix{Float64}(undef,n,num_time)
end
n = length(q)

for i = start_horizon:end_horizon
    ##########################################################################
    # sparse formulation for MIMPC:
    ##########################################################################
    if sparsity == "sparse"
        printstyled("Horizon iteration: ", i, "\n", color = :magenta)
        P, q, G,h, Ib, A, b, Ã, b̃, cones, lb, ub, i_idx= generate_sparse_MPC_Clarabel(i,0)
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
        opt_cost[i-start_horizon+1] = objective_value(model)
        opt_x[:,i-start_horizon+1] .= value.(x)[end - Nnu + 1:end]
    
    else
    ###########################################################################
    # dense formulation for MIMPC:
    ##########################################################################
        printstyled("Horizon iteration: ", i, "\n", color = :magenta)
        P, q, Ã, b̃, s, i_idx,A, b, l, u, lb, ub= generate_dense_MPC_Clarabel(i)
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
        uopt = value.(x)
        opt_cost[i-start_horizon+1] = objective_value(model)
        opt_x[:,i-start_horizon+1] .= uopt
    end
end
# println("Gurobi base_solution: ", objective_value(model2) , " using ", uopt2) 
#check_flag = all(opt_x[:,i-start_horizon+1] .== uopt2)
#@assert(check_flag == true)
#printstyled("Same solution:", check_flag, "\n",color=:green)
λ=0.99
ϵ = 1e-6

#Important data saving
iter_num_without = Matrix{Int64}(undef,num_time,repeat_time)
with_iter_num = Matrix{Int64}(undef,num_time,repeat_time)
iter_num_without_cold = Matrix{Int64}(undef,num_time,repeat_time)
with_iter_num_cold = Matrix{Int64}(undef,num_time,repeat_time)

first_iter_num = Matrix{Int64}(undef,num_time,repeat_time)
first_iter_num_cold = Matrix{Int64}(undef,num_time,repeat_time)

#total time
without_time = Matrix{Float64}(undef,num_time,repeat_time)
without_time_cold = Matrix{Float64}(undef,num_time,repeat_time)
with_time = Matrix{Float64}(undef,num_time,repeat_time)
with_time_cold = Matrix{Float64}(undef,num_time,repeat_time)

first_time_without = Matrix{Float64}(undef,num_time,repeat_time)
first_time_without_cold = Matrix{Float64}(undef,num_time,repeat_time)
first_time = Matrix{Float64}(undef,num_time,repeat_time)
first_time_cold = Matrix{Float64}(undef,num_time,repeat_time)


#fact time 
without_fact_time = Matrix{Float64}(undef,num_time,repeat_time)
without_fact_time_cold = Matrix{Float64}(undef,num_time,repeat_time)
with_fact_time = Matrix{Float64}(undef,num_time,repeat_time)
with_fact_time_cold = Matrix{Float64}(undef,num_time,repeat_time)

first_fact_time_without = Matrix{Float64}(undef,num_time,repeat_time)
first_fact_time_without_cold = Matrix{Float64}(undef,num_time,repeat_time)
first_fact_time = Matrix{Float64}(undef,num_time,repeat_time)
first_fact_time_cold = Matrix{Float64}(undef,num_time,repeat_time)    


#solve time
without_solve_time = Matrix{Float64}(undef,num_time,repeat_time)
without_solve_time_cold = Matrix{Float64}(undef,num_time,repeat_time)
with_solve_time = Matrix{Float64}(undef,num_time,repeat_time)
with_solve_time_cold = Matrix{Float64}(undef,num_time,repeat_time)

first_solve_time_without = Matrix{Float64}(undef,num_time,repeat_time)
first_solve_time_without_cold = Matrix{Float64}(undef,num_time,repeat_time)
first_solve_time = Matrix{Float64}(undef,num_time,repeat_time)
first_solve_time_cold = Matrix{Float64}(undef,num_time,repeat_time)

for t = 1:repeat_time
    for i = start_horizon:end_horizon
        ##########################################################################
        # sparse formulation for MIMPC:
        ##########################################################################
        if sparsity == "sparse"
            # printstyled("Horizon iteration: ", i, "\n", color = :magenta)
            P, q, G,h, Ib, A, b, Ã, b̃, cones, lb, ub, i_idx= generate_sparse_MPC_Clarabel(i,0)
            n = length(q)
            Nnu = length(i_idx)

        else
        ###########################################################################
        # dense formulation for MIMPC:
        ##########################################################################
            # printstyled("Horizon iteration: ", i, "\n", color = :magenta)
            P, q, Ã, b̃, cones, i_idx,A, b, l, u, lb, ub= generate_dense_MPC_Clarabel(i)
            n = length(q)
        end

        λ=0.99
        η= 1000 # penalisation for y_B
        γ = 100 # penalisation for z
        σ = 1e-7 # perturbation added to P in optimization matrix

        ϵ = 1e-6
    
        settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
        # #################################################
        # # YC: solve the problem with early termination with warm_start
        # #################################################
        # solver   = Clarabel.Solver()
        # printstyled("with early termination with warm_start: \n", color = :green)
        # Clarabel.setup!(solver, P, q, Ã, b̃,cones, settings)
        # base_solution = Clarabel.solve!(solver)
        # # println("Clarabel base result " ,base_solution, " with base_solution ", base_solution.x[end - Nnu + 1:end])

        # # #start bnb loop
        # # println("STARTING CLARABEL BNB LOOP ")

        # early_termination = Int(1)          #choose our proposed optimization-based early termination
        # best_ub, feasible_solution, early_num, total_iter, fea_iter, total_time, fea_time, total_fact_time, fea_fact_time, total_solve_time, fea_solve_time, total_nodes,fea_nodes = branch_and_bound_solve(i, solver, base_solution,n,ϵ, i_idx, true, early_termination, warm_start, λ,luS,debug_print,false) 

        # diff_sol_vector = feasible_solution - opt_x[:,i-start_horizon+1]

        # diff_solution=round(norm(diff_sol_vector),digits=3)
        # diff_obj = round(best_ub-opt_cost[i-start_horizon+1],digits=6)
        # @assert(diff_obj ≈ 0)       # YC: Check whether converge to the same optimum
        # if ~iszero(diff_solution) || ~iszero(diff_obj)
        #     println("Solution diff: ",diff_solution, "Obj difference: ", diff_obj)
        #     println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
        #     println("Horizon iteration number: ", i)
        #     error("Solutions differ!")
        #     #num_errors = num_errors +1
        # end

        #################################################
        # YC: solve the problem with early termination with cold start
        #################################################
        solver_cold  = Clarabel.Solver()
        # printstyled("with early termination with cold_start: \n", color = :green)
        Clarabel.setup!(solver_cold, P, q, Ã, b̃,cones, settings)
        base_solution_cold = Clarabel.solve!(solver_cold)
        # println("Clarabel base result " ,base_solution, " with base_solution ", base_solution.x[end - Nnu + 1:end])

        # #start bnb loop
        # println("STARTING CLARABEL BNB LOOP ")

        if sparsity == "sparse"
            luS = factorize_optimization_based_matrix(solver_cold.data,Ib,G,σ,η, γ)
        else
            luS = nothing
        end

        # if sparsity == "sparse"
        #     early_termination_cold = Int(2)     #optimization correction for the sparse form
        # else
        #     early_termination_cold = Int(1)     #Simple correction for the dense form
        # end
        early_termination_cold = true
        best_ub_cold, feasible_solution_cold, early_num_cold, total_iter_cold, fea_iter_cold, total_time_cold, fea_time_cold, total_fact_time_cold, fea_fact_time_cold, total_solve_time_cold, fea_solve_time_cold, total_nodes_cold,fea_nodes_cold = branch_and_bound_solve(i, solver_cold, base_solution_cold,n,ϵ, i_idx, true, early_termination_cold, warm_start, λ,luS,debug_print,false) 

        if sparsity == "sparse"
            diff_sol_vector_cold = feasible_solution_cold[end - Nnu + 1:end] - opt_x[:,i-start_horizon+1]
        else
            diff_sol_vector_cold = feasible_solution_cold - opt_x[:,i-start_horizon+1]
        end

        diff_solution_cold=round(norm(diff_sol_vector_cold),digits=3)
        diff_obj_cold = round(best_ub_cold-opt_cost[i-start_horizon+1],digits=5)
        @assert(diff_obj_cold ≈ 0)       # YC: Check whether converge to the same optimum
        
        
        # #################################################
        # # YC: solve the problem without early termination and with warm start
        # #################################################
        # solver_without   = Clarabel.Solver()
        # # printstyled("no early termination with warm_start: \n", color = :green)
        # Clarabel.setup!(solver_without, P, q, Ã, b̃,cones, settings)
        
        # base_solution_without = Clarabel.solve!(solver_without)

        # early_termination = Int(0)
        # best_ub_without, feasible_solution_without, early_num_without, total_iter_without, fea_iter_without, total_time_without, fea_time_without, total_fact_time_without, fea_fact_time_without, total_solve_time_without, fea_solve_time_without, total_nodes_without,fea_nodes_without = branch_and_bound_solve(i,solver_without, base_solution_without,n,ϵ, i_idx, true, early_termination, warm_start,λ,luS,debug_print,false) 
        
        # # diff_sol_vector .= feasible_solution_without[end - Nnu + 1:end] - opt_x[:,i-start_horizon+1]
        # diff_sol_vector= feasible_solution_without - opt_x[:,i-start_horizon+1]
        
        # diff_solution=round(norm(diff_sol_vector),digits=3)
        # diff_obj = round(best_ub_without-opt_cost[i-start_horizon+1],digits=6)
        # @assert(diff_obj ≈ 0) 
        # @assert(total_iter_without - fea_iter_without >= 0.0)

        #################################################
        # YC: solve the problem without early termination and with cold start
        #################################################
        solver_without_cold   = Clarabel.Solver()
        # printstyled("without early termination with cold_start: \n", color = :green)
        Clarabel.setup!(solver_without_cold, P, q, Ã, b̃,cones, settings)
        
        base_solution_without_cold = Clarabel.solve!(solver_without_cold)

        early_termination = false
        best_ub_without_cold, feasible_solution_without_cold, early_num_without_cold, total_iter_without_cold, fea_iter_without_cold, total_time_without_cold, fea_time_without_cold, total_fact_time_without_cold, fea_fact_time_without_cold, total_solve_time_without_cold, fea_solve_time_without_cold, total_nodes_without_cold,fea_nodes_without_cold = branch_and_bound_solve(i,solver_without_cold, base_solution_without_cold,n,ϵ, i_idx, true, early_termination, warm_start,λ,luS,debug_print,false) 
        
        if sparsity == "sparse"
            diff_sol_vector_cold .= feasible_solution_without_cold[end - Nnu + 1:end] - opt_x[:,i-start_horizon+1]
        else
            diff_sol_vector_cold .= feasible_solution_without_cold - opt_x[:,i-start_horizon+1]
        end
        
        diff_solution=round(norm(diff_sol_vector_cold),digits=3)
        diff_obj = round(best_ub_without_cold-opt_cost[i-start_horizon+1],digits=6)
        @assert(diff_obj ≈ 0) 
        @assert(total_iter_without_cold - fea_iter_without_cold >= 0.0)


        # @assert total_nodes == total_nodes_without
        # @assert total_nodes == total_nodes_without_cold
        @assert total_nodes_cold == total_nodes_without_cold

        #Store information for each time interval
        # iter_num_without[i - start_horizon + 1,t] = total_iter_without
        # with_iter_num[i - start_horizon + 1,t] = total_iter
        iter_num_without_cold[i - start_horizon + 1,t] = total_iter_without_cold
        with_iter_num_cold[i - start_horizon + 1,t] = total_iter_cold

        @assert(fea_iter_cold == fea_iter_without_cold)
        first_iter_num_cold[i - start_horizon + 1,t] = fea_iter_cold

        #total time            
        # save_time(i, t, start_horizon, first_time, with_time, fea_time, total_time)
        save_time(i, t, start_horizon, first_time_cold, with_time_cold, fea_time_cold, total_time_cold)
        # save_time(i, t, start_horizon, first_time_without, without_time, fea_time_without, total_time_without)
        save_time(i, t, start_horizon, first_time_without_cold, without_time_cold, fea_time_without_cold, total_time_without_cold)
        
        #fact time
        # save_time(i, t, start_horizon, first_fact_time, with_fact_time, fea_fact_time, total_fact_time)
        save_time(i, t, start_horizon, first_fact_time_cold, with_fact_time_cold, fea_fact_time_cold, total_fact_time_cold)
        # save_time(i, t, start_horizon, first_fact_time_without, without_fact_time, fea_fact_time_without, total_fact_time_without)
        save_time(i, t, start_horizon, first_fact_time_without_cold, without_fact_time_cold, fea_fact_time_without_cold, total_fact_time_without_cold)

        # #solve time
        # save_time(i, t, start_horizon, first_solve_time, with_solve_time, fea_solve_time, total_solve_time)
        save_time(i, t, start_horizon, first_solve_time_cold, with_solve_time_cold, fea_solve_time_cold, total_solve_time_cold)
        # save_time(i, t, start_horizon, first_solve_time_without, without_solve_time, fea_solve_time_without, total_solve_time_without)
        save_time(i, t, start_horizon, first_solve_time_without_cold, without_solve_time_cold, fea_solve_time_without_cold, total_solve_time_without_cold)

    end
end

#Averaging data
iter_num_without_cold = mean(iter_num_without_cold, dims=2)
with_iter_num_cold = mean(with_iter_num_cold, dims=2)
first_iter_num_cold = mean(first_iter_num_cold, dims=2)

#Averaging time
# first_time_without = mean(first_time_without, dims=2)
first_time_without_cold = mean(first_time_without_cold, dims=2)
# first_time = mean(first_time, dims=2)
first_time_cold = mean(first_time_cold, dims=2)

# without_time = mean(without_time, dims = 2)
without_time_cold = mean(without_time_cold, dims = 2)
# with_time = mean(with_time, dims = 2)
with_time_cold = mean(with_time_cold, dims = 2)

#fact_time 
# first_fact_time_without = mean(first_fact_time_without, dims=2)
first_fact_time_without_cold = mean(first_fact_time_without_cold, dims=2)
# first_fact_time = mean(first_fact_time, dims=2)
first_fact_time_cold = mean(first_fact_time_cold, dims=2)

# without_fact_time = mean(without_fact_time, dims = 2)
without_fact_time_cold = mean(without_fact_time_cold, dims = 2)
# with_fact_time = mean(with_fact_time, dims = 2)
with_fact_time_cold = mean(with_fact_time_cold, dims = 2)

#solve_time
# first_solve_time_without = mean(first_solve_time_without, dims=2)
first_solve_time_without_cold = mean(first_solve_time_without_cold, dims=2)
# first_solve_time = mean(first_solve_time, dims=2)
first_solve_time_cold = mean(first_solve_time_cold, dims=2)

# without_solve_time = mean(without_solve_time, dims = 2)
without_solve_time_cold = mean(without_solve_time_cold, dims = 2)
# with_solve_time = mean(with_solve_time, dims = 2)
with_solve_time_cold = mean(with_solve_time_cold, dims = 2)

################################################################
# Plot ratio reduction
################################################################

# percentage = (with_iter_num_cold .- first_iter_num_cold) ./(iter_num_without_cold .- first_iter_num_cold)
# for i = 1:lastindex(percentage)
#     if isnan(percentage[i])
#         percentage[i] = 1.0
#     end
# end

# #early termination with warm start
# percentage_with_time = (with_time .- first_time) ./(without_time_cold .- first_time_without_cold)
# for i = 1:lastindex(percentage_with_time)
#     if isnan(percentage_with_time[i]) || isinf(percentage_with_time[i])
#         percentage_with_time[i] = 1.0
#     end
# end

#early termination with cold start
# @assert(all(with_time_cold .- first_time_cold .>= 0.0))
total_time_cold = (with_time_cold .- first_time_cold)*1e3           #change the metric from s to ms
# @assert(all(without_time_cold .- first_time_without_cold .>= 0.0))
total_time_without_cold = (without_time_cold .- first_time_without_cold)*1e3
percentage_with_time_cold = total_time_cold ./total_time_without_cold
for i = 1:lastindex(percentage_with_time_cold)
    if isnan(percentage_with_time_cold[i]) || isinf(percentage_with_time_cold[i])
        percentage_with_time_cold[i] = 1.0
    end
end

# #no early termination with warm start
# percentage_without_time = (without_time .- first_time_without) ./(without_time_cold .- first_time_without_cold)
# for i = 1:lastindex(percentage_without_time)
#     if isnan(percentage_without_time[i]) || isinf(percentage_without_time[i])
#         percentage_without_time[i] = 1.0
#     end
# end

percentage_without_time_cold = ones(length(percentage_with_time_cold))

ind = 0:(end_horizon - start_horizon)

save(@sprintf("mpc_%s_N=%d.jld",sparsity,N), "percentage_ratio", percentage_with_time_cold, 
                                                "total_time", total_time_cold,
                                                "total_time_without", total_time_without_cold,
                                                "ind", ind
                                                )

# Plots.plot(ind,[percentage_with_time_cold percentage_without_time_cold], seriestype=[:steppost :steppost], label = ["Early termination" "No early termination"], color = ["red" "black"])
# # Plots.plot(ind,[percentage percentage_time], seriestype=[:steppost :steppost], label = ["iter" "total_time"], color = ["black" "red"])

# ylabel!("Ratio")
# xlabel!("Intervals")
# savefig(@sprintf("yc_mpc_time_%s.pdf",sparsity))
