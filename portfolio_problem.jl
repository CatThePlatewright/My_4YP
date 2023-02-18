include("toy_bnb.jl")
using JLD

function create_bounded_variables(model,n,L)
    x = @variable(model, 0 <= x[i = 1:total_num] <= 1.0)
    xplus = x[1:n]
    xminus = x[n+1:2*n]
    ζ =  x[2*n+1:2*n+L]
    ρ = x[end:end]
    set_lower_bound.(x[1:total_num-1],0) # ρ is free
    set_upper_bound.(x[1:total_num-1],1.0)
    return x, xplus, xminus, ζ, ρ
end
function add_constraints(model,n,L,Σ,w, smin)
    #cost_con_select_vars_matrix = [Matrix(1I,n,n) Matrix(1I,n,n) zeros(n,n) zeros(n,1)]
    #@constraint(model, cost_constraint, [ρ; Σ^0.5*cost_con_select_vars_matrix*x] in SecondOrderCone())
    @constraint(model, cost_constraint, [1+ρ; [Σ^0.5*(xplus+xminus); 1-ρ]] in SecondOrderCone()) #TODO: cannot mix and match the ρ or ̢ρ squared 
    @constraint(model, sum_constraint, sum(w+xplus-xminus) == 1.0)
    @constraint(model, sufficient_amount_con, smin*ζ .<= sum(w+xplus-xminus)*ones(length(ζ)).<=smin+(1-smin)*ζ)
    
end
function getData(n,L, sum_of_bin_vars)
    total_num = 2*n+L+1
    Random.seed!(1234)
    γ = rand(Float64,n) # expected return rates
    q = vcat(-γ,γ,zeros(L),0)
    P = sparse([total_num],[total_num],[1.0],total_num,total_num)
    binary_vars = collect(2*n+1:2*n+L) # indices of binary variables
    
    # check against Gurobi's mixed-integer solution
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)
    x, xplus, xminus, ζ, ρ = create_bounded_variables(exact_model,n,L)
    set_binary.(ζ)

    @objective(exact_model, Min, 0.5*ρ^2 + q'*x)
    @objective(exact_model, Min, ρ + q'*x)
    add_constraints(exact_model,n,L,Σ,w,smin)
    optimize!(exact_model)
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 

    # create Clarabel instance, solve the fully relaxed base problem
    optimizer = Clarabel.Optimizer
    model = Model()
    setOptimizer(model, optimizer)
    x, xplus, xminus, ζ, ρ = create_bounded_variables(model,n,L)
    @objective(exact_model, Min, 0.5*ρ^2 + q'*x)
    add_constraints(exact_model,n,L,Σ,w,smin)
    solve_base_model(model)    #solve in Clarabel the relaxed problem
    P,q,A,b, cones= getClarabelData(model)
    println("P: ", P)
    println("q : ", q)
    println("A : ", A)
    println("b : ", b)
    println("cones : ", cones) 
    println("integer vars: ", binary_vars)
    return P,q,A,b,cones, binary_vars, exact_model
    
end

n_range =4:4
k= 2
λ = 0.99
η = 100.0
ϵ = 1e-6
without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]
percentage_iter_reduction = Float64[]

for n in n_range
    P,q,A,b, s, binary_vars, exact_model= getData(n,k)
    simple_domain_propagation!(b,k)
    println("Domain propagated b: ", b)
    println("Setting up Clarabel solver...")
    settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, A, b, s, settings)

    result = Clarabel.solve!(solver, Inf)


    #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")
 
    best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(solver, result,n,ϵ, binary_vars,true,true,true,λ,η) 
    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
    diff_sol_vector= feasible_solution - value.(exact_model[:x])
    diff_solution=round(norm(diff_sol_vector),digits=5)
    diff_obj = round(best_ub-objective_value(exact_model),digits=5)
    if ~iszero(diff_solution) || ~iszero(diff_obj)
        println("Solution diff: ",diff_solution, "Obj difference: ", diff_obj)
        println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
        error("Solutions differ!")
    end
    println("Number of early terminated nodes: ", early_num)

    # count QP iterations
    printstyled("Total net iter num (with early_term_enable): ", total_iter-fea_iter, "\n", color = :green)
    println(" ")  
#=     solver_without   = Clarabel.Solver()

    Clarabel.setup!(solver_without, P, q, A, b,s, settings)

    base_solution_without = Clarabel.solve!(solver_without)
    best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(solver_without, base_solution_without,n,ϵ, binary_vars, true, false, false,λ) 
    println("Found objective without early_term: ", best_ub_without)
    printstyled("Total net iter num (without): ", total_iter_without - fea_iter_without, "\n", color = :green)

    println("Number of early terminated nodes (without): ", early_num_without)
    reduction = 1 - (total_iter - fea_iter)/(total_iter_without - fea_iter_without)
    println("Reduced iterations (percentage): ", reduction)
    append!(without_iter_num, total_iter_without)
    append!(with_iter_num, total_iter)
    append!(percentage_iter_reduction, reduction)
    if (fea_iter == fea_iter_without)
        append!(first_iter_num, fea_iter)
    end   =#
end


printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)
#save("my_toy_k=10_warmstart.jld", "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction)
