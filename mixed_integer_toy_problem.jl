include("mixed_integer_toy_bnb.jl")
using JLD
function swaporder(A,b,n)
    A = Matrix(A)
    new_A = vcat(A[1:1,:], A[2*n+2:end-2*n,:], A[2:2*n+1,:], A[end-2*n+1:end,:])
    new_b = vcat(b[1], b[2*n+2:end-2*n], b[2:2*n+1], b[end-2*n+1:end])
    return sparse(new_A), new_b
end
function getData(n,sum_of_bin_vars)
    Q = Matrix{Float64}(I, n, n) 
    Random.seed!(1234)
    c = rand(Float64,n)*2 .-1
    
    # check against Gurobi's mixed-integer solution
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)

    m = 2*n #total number of variables, n continuous x's, n binary vars
    z = @variable(exact_model, x[i = 1:m])
    x = z[1:n]
    binary_vars = collect(n+1:m) # indices of binary variables
    y= z[n+1:m]
    set_binary.(y)
    set_lower_bound.(x,-10)
    set_upper_bound.(x,10)

    @objective(exact_model, Min, x'*Q*x + c'*x)
    @constraint(exact_model, sum_constraint, sum(y) == sum_of_bin_vars)
    @constraint(exact_model, x.<=10*y)
    @constraint(exact_model, -10*y .<= x)
    optimize!(exact_model)
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 

    # solve the fully relaxed base problem
    optimizer = Clarabel.Optimizer
    model = Model()
    setOptimizer(model, optimizer)
    z = @variable(model, x[1:m])
    x = z[1:n]
    y= z[n+1:m]
    @objective(model, Min, x'*Q*x + c'*x)
    @constraint(model, sum_constraint, sum(y) == sum_of_bin_vars)
    @constraint(model, x.<=10*y)
    @constraint(model, -10*y .<= x)
    
    @constraint(model, lbx, x.>= -10*ones(n)) #try .<= instead 
    @constraint(model, lby, y.>= zeros(n))
    @constraint(model, ubx, x.<= 10*ones(n))
    @constraint(model, uby, y.<= ones(n))

    solve_base_model(model)    #solve in Clarabel the relaxed problem
    P,q,A,b, cones= getClarabelData(model)
    #A, b = swaporder(A,b,n)
    println("P: ", P)
    println("q : ", q)
    println("A : ", A)
    println("b : ", b)
    println("cones : ", cones) 
    println("integer vars: ", binary_vars)
    return P,q,A,b,cones, binary_vars, exact_model
    
end

n_range =2:2
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
 
    best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(solver, result,2*n,ϵ, binary_vars,false,false,false,λ,η) #want total number of vars: 2*n
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
    solver_without   = Clarabel.Solver()

    Clarabel.setup!(solver_without, P, q, A, b,s, settings)

    base_solution_without = Clarabel.solve!(solver_without)
    best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(solver_without, base_solution_without,n,ϵ, binary_vars, false, false, false,λ) 
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
    end  
end


printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)
#save("my_toy_k=10_warmstart.jld", "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction)
