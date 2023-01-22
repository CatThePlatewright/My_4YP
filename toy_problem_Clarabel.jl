include("toy_bnb.jl")
using JLD

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

n_range = 5:15
k_range = -1:4
λ = 0.99
ϵ = 0.00000001
without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]
percentage_iter_reduction = Float64[]

for n in n_range
    for k in k_range
        m=n
        P,q,A,b, s, integer_vars, exact_model= getData(n,m,k)
        #Ā,b̄, s̄= getAugmentedData(A,b,cones,integer_vars,n) we do not want to augment them anymore...
        simple_domain_propagation!(b,k)
        println("Domain propagated b: ", b)
        println("Setting up Clarabel solver...")
        settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
        solver   = Clarabel.Solver()

        Clarabel.setup!(solver, P, q, A, b, s, settings)

        result = Clarabel.solve!(solver, Inf)


        #start bnb loop
        println("STARTING CLARABEL BNB LOOP ")

        time_taken = @elapsed begin
            best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(solver, result,n,ϵ, integer_vars,true,true,true,λ) 
        end
        println("Time taken by bnb loop: ", time_taken)
        println("Termination status of Clarabel solver:" , solver.info.status)
        println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
        diff_sol_vector= feasible_solution - value.(exact_model[:x])
        diff_solution=round(norm(diff_sol_vector),digits=5)
        diff_obj = round(best_ub-objective_value(exact_model),digits=6)
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
        best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(solver_without, base_solution_without,n,ϵ, integer_vars, true, true, false,λ) 
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
    end
end


printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)
save("toy_problem.jld", "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction)
