include("mixed_binary_solver.jl")
using Test, ECOS
# imported functions from mixed_binary_solver:
# add_constraints, fix_variables(), build_unbounded_base_model()

optimizer = Clarabel.Optimizer
n = 5
k= 3
m =3 # how many integer variables (if mixed integer problem)
integer_vars = sample(1:n, m, replace = false)
sort!(integer_vars) 
println("Integer variables : ", integer_vars)
println("CAREFUL: if comparing with getClarabel.jl, integer_vars may be different sample!!!")
Q = Matrix{Float64}(I, n, n) 
Random.seed!(1234)
c = rand(Float64,n)
ϵ = 0.00000001

function main()
    

    # check against binary solver in Gurobi
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)
    x = @variable(exact_model, x[i = 1:n])
    for bin in integer_vars
        set_integer(x[bin])
    end
    @objective(exact_model, Min, x'*Q*x + c'*x)
    @constraint(exact_model, sum_constraint, sum(x) == k)
    optimize!(exact_model)
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 


    
    base_model = build_unbounded_base_model(optimizer,n,k,Q,c)

    solve_base_model(base_model,integer_vars)

    root, term_status = branch_and_bound_solve(base_model,optimizer,n,ϵ, integer_vars)
    @test term_status == "OPTIMAL"
    println("Found objective: ", root.data.ub, " using ", root.data.solution_x)


    tol = 1e-4

    @test isapprox(norm(root.data.solution_x - Float64.(value.(exact_model[:x]))), zero(Float64), atol=tol)
    @test isapprox(root.data.ub, Float64(objective_value(exact_model)), atol=tol)
        
end