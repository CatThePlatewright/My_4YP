include("portfolio_bnb.jl")
using JLD, CSV, Statistics, LinearAlgebra,Printf

function create_variables(model,N,L)
    @variable(model,xplus[i = 1:N])
    @variable(model,xminus[i = 1:N])
    @variable(model,bin[i = 1:N])
    @variable(model,l[i=1:L])

end
function add_constraints(model,N,L, M,K,Lmin, Lmax,ρ,Λ)

    @constraint(model, cost_constraint, [1+ρ; [sqrt(Λ)*(model[:xplus]-model[:xminus]); 1-ρ]] in SecondOrderCone()) 
    # sum constraints
    @constraint(model, sum_investments, sum(model[:xplus]-model[:xminus]) == M)
    @constraint(model, sum_number_investments, sum(model[:bin]) <= K)
    @constraint(model, sum_sectors_lb, -sum(model[:l]) <= -Lmin)
    @constraint(model, sum_sectors_ub, sum(model[:l]) <= Lmax)
    # sector-asset pair constraints
    @constraint(model, sector_asset1, model[:bin] .<= H*model[:l])# if you invest into ith asset, the corresponding sector must be chosen, took
    @constraint(model,sector_asset2, model[:l] .<= H'*model[:bin])# if you choose jth sector, there must be investment into the corresponding assets
    # lower bounds on individual variables
    @constraint(model,lxplus,-model[:xplus] .<=zeros(N))
    @constraint(model,lxminus,-model[:xminus] .<=zeros(N))
    @constraint(model,lbin,-model[:bin] .<=zeros(N))
    @constraint(model,ll,-model[:l] .<=zeros(L))
    # upper bounds on individual variables
    @constraint(model,uxplus,model[:xplus] .<= M*model[:bin])
    @constraint(model,uxminus,model[:xminus] .<= M*model[:bin])
    @constraint(model,ubin,model[:bin] .<=ones(N))
    @constraint(model,ul,model[:l] .<=ones(L))
    
end

"""returns the nxn covariance matrix between the returns of n assets,
stored in R, a matrix of size Txn where T is the number of days in the data (2516)"""
function calc_variance(N::Int, R::Matrix) 
    Λ = zeros(N,N)
    for i = 1:N
        for j = 1:N
            Λ[i,j] = cov(R[:,i], R[:,j]) # average over the time frame day 0 to day 2516
        end
    end
    return Λ
end

function get_return_data(N::Int, T::Int)
    returns = CSV.File("portfolio_returns.csv")
    returns = returns[1:T] # just get first row of returns at time t=0
    R = zeros(T,N)
    companies = propertynames(returns[1])
    for row in 1:lastindex(returns)
        for col in 1:lastindex(companies)-1
            if col <= N
                R[row,col]= getproperty(returns[row],companies[col+1])
            end
        end
    end
    return R
end
function get_sector_asset_matrix()
    asset2sector = [1,1,1,3,2,3,2,1,3,3,2,2,3,2,3,2,1,2,1,1]
    H = sparse(collect(1:N),asset2sector[1:N],ones(N))
    return H
end
function solveGurobi(ρ,r, Λ)
    # check against Gurobi's mixed-integer solution
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)
    create_variables(exact_model,N,L) 
    set_binary.(exact_model[:bin])
    set_binary.(exact_model[:l])
    @objective(exact_model, Min, -sum(r[i]*(exact_model[:xplus][i] -exact_model[:xminus][i]) for i = 1:N))
    add_constraints(exact_model,N,L,M,K,Lmin,Lmax,ρ,Λ)
    println(exact_model)

    optimize!(exact_model)
    solution = vcat(value.(exact_model[:xplus]), value.(exact_model[:xminus]), value.(exact_model[:bin]), value.(exact_model[:l]))
    println("Exact (Gurobi) solution: ", objective_value(exact_model) , " using ", solution) 

    return exact_model,solution
end
function getData(ρ,r,Λ)
    
    # create Clarabel instance, solve the fully relaxed base problem
    optimizer = Clarabel.Optimizer
    model = Model()
    setOptimizer(model, optimizer)
    create_variables(model,N,L)
    @objective(model,Min,-sum(r[i]*(model[:xplus][i] -model[:xminus][i]) for i = 1:N)) # maximise average return (constant 1/T not included here)
    add_constraints(model,N,L, M,K,Lmin, Lmax, ρ,Λ)
    optimize!(model)
    print("Solve_base_model in JuMP: ",solution_summary(model))
    println("JuMP solution relaxed: ",  objective_value(model) , " using ", value.(model[:xplus]), value.(model[:xminus]), value.(model[:bin]), value.(model[:l]))
    P,q,A,b, cones= getClarabelData(model)
    #=println("A : ", A)
    println("b : ", b)
    println("cones : ", cones) 
    println("integer vars: ", binary_vars)=#
    return P,q,A,b,cones, binary_vars
    
end
# Cumulative portfolo value over time
function portfolio_V(V_0, r)
    return V_0 * cumprod(1 .+ r,dims=1)
end 

N = 20 # 20 is max. number of assets
L = 3
M = 1.0 # total investement (money)
K = 2 # maximum number of investments (sum of bin vars)
Lmin = 1
Lmax = L
λ = 0.99
ϵ = 1e-6
total_num = 3*N+L
T = 2000 # end of period (how many days to consider / rows in data sheet)
R = get_return_data(N,T)# return rates
Λ = calc_variance(N, R)
H = get_sector_asset_matrix()
binary_vars = collect(2*N+1:3*N+L) # indices of binary variables
without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]
percentage_iter_reduction = Float64[]
total_nodes_num = Int64[]
total_nodes_without_num = Int64[]
asset_distribution= Float64[]

V0 = 10000
ρ_values = [1]# SOC constraint for risk return
ρ_values_str = ["1"]
#= w = zeros(20,1)
w[20]=1.0
r=R*w =#

for i in 1:lastindex(ρ_values)
    ρ = ρ_values[i]
    r = (1/T)*ones(1,T)*R
    exact_model, exact_solution = solveGurobi(ρ,r,Λ)
    P,q,A,b, s, binary_vars= getData(ρ,r,Λ)
    println("Setting up Clarabel solver...")
    settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, A, b, s, settings)

    result = Clarabel.solve!(solver, Inf)
    #println(result)

    #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")
    best_ub, feasible_solution, early_num, total_iter, fea_iter, total_nodes = branch_and_bound_solve(solver, result,N,L,ϵ, binary_vars,true,true,false,λ) #want total number of vars: 2*n

    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
    x_plus = feasible_solution[1:N]
    x_minus = feasible_solution[N+1:2*N]
    r_solution = R*(x_plus-x_minus) 
    diff_sol_vector= round.(feasible_solution - value.(exact_solution),digits=5)
    diff_solution=round(norm(diff_sol_vector),digits=4)
    diff_obj = round(best_ub-objective_value(exact_model),digits=4) # digits=5 resulted in one different solution by 2.0e-5
    if ~iszero(diff_obj)
        println("Solution diff: ",diff_solution, "--- Objective difference: ", diff_obj)
        println(diff_sol_vector)
        println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
        error("Solutions differ!")
    end
    
    portfolio_value = portfolio_V(V0,r_solution)

    save(@sprintf("portfolio_%s.jld",ρ_values_str[i]), "Vt",portfolio_value,"xplus",x_plus,"xminus",x_minus,"r_solution",r_solution)

    
end

for i in 1:lastindex(ρ_values)
    ρ = ρ_values[i]
    for t in 1000:1200
        R = get_return_data(N,t) # return rates
        Λ = calc_variance(N, R)
        r = (1/t)*ones(1,t)*R

        exact_model, exact_solution = solveGurobi(ρ,r,Λ)
        P,q,A,b, s, binary_vars= getData(ρ,r,Λ)
        println("Setting up Clarabel solver...")
        settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
        solver   = Clarabel.Solver()

        Clarabel.setup!(solver, P, q, A, b, s, settings)

        result = Clarabel.solve!(solver, Inf)
        println(result)

        #start bnb loop
        println("STARTING CLARABEL BNB LOOP ")
        best_ub, feasible_solution, early_num, total_iter, fea_iter, total_nodes = branch_and_bound_solve(solver, result,N,L,ϵ, binary_vars,true,true,false,λ) #want total number of vars: 2*n

        println("Termination status of Clarabel solver:" , solver.info.status)
        println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
        diff_sol_vector= round.(feasible_solution - value.(exact_solution),digits=5)
        diff_solution=round(norm(diff_sol_vector),digits=5)
        diff_obj = round(best_ub-objective_value(exact_model),digits=5)
        if ~iszero(diff_obj)
            println("Solution diff: ",diff_solution, "--- Objective difference: ", diff_obj)
            println(diff_sol_vector)
            println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
            error("Solutions differ!")
        end
        
        println("Number of early terminated nodes: ", early_num)

        # count QP iterations
        printstyled("Total net iter num (with early_term_enable): ", total_iter-fea_iter, "\n", color = :green)
        println(" ")  


        solver_without   = Clarabel.Solver()

        Clarabel.setup!(solver_without, P, q, A, b,s, settings)

        base_solution_without = Clarabel.solve!(solver_without)
        best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without, total_nodes_without = branch_and_bound_solve(solver_without, base_solution_without,N,L,ϵ, binary_vars, true, false, false,λ) 
        println("Found objective without early_term: ", best_ub_without)
        printstyled("Total net iter num (without): ", total_iter_without - fea_iter_without, "\n", color = :green)

        println("Number of early terminated nodes (without): ", early_num_without)
        reduction = 1 - (total_iter - fea_iter)/(total_iter_without - fea_iter_without)
        println("Reduced iterations (percentage): ", reduction)
        append!(without_iter_num, total_iter_without)
        append!(with_iter_num, total_iter)
        append!(percentage_iter_reduction, reduction)
        append!(total_nodes_num, total_nodes)
        append!(total_nodes_without_num, total_nodes_without)
        if (fea_iter == fea_iter_without)
            append!(first_iter_num, fea_iter)
        end  
    end
    save(@sprintf("portfolio_iterations_%s.jld",ρ_values_str[i]), "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction, "total_nodes", total_nodes_num, "total_nodes_without", total_nodes_without_num)

end 

printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)
