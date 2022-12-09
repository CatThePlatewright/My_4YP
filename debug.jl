include("direct_Clarabel_one_solver.jl")
n = 5
k = 7
m= 5
integer_vars = sample(1:n, m, replace = false)
sort!(integer_vars)
Q = Matrix{Float64}(I, n, n) 
Random.seed!(1234)
c = rand(Float64,n)
ϵ = 0.00000001
    # check against Gurobi
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

optimizer = Clarabel.Optimizer
old_model = build_unbounded_base_model(optimizer,n,k,Q,c)
solve_base_model(old_model,integer_vars)

#solve in Clarabel the relaxed problem


P,q,A,b, cones= getClarabelData(old_model)
Ā,b̄, s̄= getAugmentedData(A,b,cones,integer_vars,n)
settings = Clarabel.Settings(verbose = true, equilibrate_enable = false, max_iter = 100)
solver   = Clarabel.Solver()

Clarabel.setup!(solver, P, q, Ā, b̄, s̄, settings)
println(" A : ",solver.data.A)
println(" b ", solver.data.b)

result = Clarabel.solve!(solver)


#start bnb loop
println("STARTING CLARABEL BNB LOOP ")
root, term_status = branch_and_bound_solve(solver, result,n,ϵ, integer_vars)
println("Termination status of Clarabel bnb:" , term_status)
println("Found objective: ", root.data.ub, " using ", round.(root.data.solution_x,digits=3))
println("Compare with exact: ", round(norm(root.data.solution_x - value.(exact_model[:x]))),round(root.data.ub-objective_value(exact_model)))
println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 

