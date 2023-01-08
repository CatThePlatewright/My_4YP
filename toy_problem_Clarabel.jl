include("direct_Clarabel_large_augmented.jl")
function getData(n,m,k)
    integer_vars = sample(1:n, m, replace = false)
    sort!(integer_vars)
    Q = Matrix{Float64}(I, n, n) 
    Random.seed!(1234)
    c = rand(Float64,n)
    
    # check against Gurobi's mixed-integer solution
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)
    x = @variable(exact_model, x[i = 1:n])
    for i in integer_vars
        set_integer(x[i])
    end
    @objective(exact_model, Min, x'*Q*x + c'*x)
    @constraint(exact_model, sum_constraint, sum(x) == k)
    optimize!(exact_model)
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 

    # solve the fully relaxed base problem
    optimizer = Clarabel.Optimizer
    old_model = build_unbounded_base_model(optimizer,n,k,Q,c)
    solve_base_model(old_model,integer_vars)    #solve in Clarabel the relaxed problem
    P,q,A,b, cones= getClarabelData(old_model)
    println("P: ", P)
    println("q : ", q)
    println("A : ", A)
    println("b : ", b)
    println("cones : ", cones) 
    return P,q,A,b,cones, integer_vars, exact_model
    
end
function main_Clarabel()
    n = 2
    m = 2
    k = 3
    ϵ = 0.00000001

    P,q,Ā,b̄, s̄, integer_vars, exact_model= getData(n,m,k)
    #Ā,b̄, s̄= getAugmentedData(A,b,cones,integer_vars,n)
    simple_domain_propagation_4N_augmented!(b̄,k)
    println("Domain propagated b: ", b̄)
    println("Setting up Clarabel solver...")
    settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ā, b̄, s̄, settings)
    
    result = Clarabel.solve!(solver, Inf)


    #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")

    time_taken = @elapsed begin
     best_ub, feasible_solution, early_num = branch_and_bound_solve(solver, result,n,ϵ, integer_vars) 
    end
    println("Time taken by bnb loop: ", time_taken)
    println("Termination status of Clarabel solver:" , solver.info.status)
    println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
    println("Compare with exact: ", round(norm(feasible_solution - value.(exact_model[:x]))),round(best_ub-objective_value(exact_model)))
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 
    println("Number of early terminated nodes: ", early_num)
    
    return solver
end

main_Clarabel()