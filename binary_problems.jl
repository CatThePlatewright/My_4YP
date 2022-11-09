include("mixed_binary_solver.jl")
FloatT = Float64
optimizer = Gurobi.Optimizer
n = 4
m = 4 # how many binary variables
Random.seed!(1234)
A = rand(n,n)
A = A'*A
b = rand(FloatT,n)
ϵ = 0.00000001
function build_unbounded_cardinality_model(optimizer, n::Int,A::Matrix,b::Vector)
    model = Model()
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    @variable(model, x[1:n])
    @objective(model, Min, sum(x.^2))
    @constraint(model, A*x .== b) 
    return model
end
base_model = build_unbounded_cardinality_model(optimizer,n,A,b)
binary_vars = sample(1:n, m, replace = false)
sort!(binary_vars)
root = branch_and_bound_solve(base_model,optimizer,n,ϵ, binary_vars)
@test termination_status(root.data.model)   == OPTIMAL
println("Found objective: ", root.data.ub, " using ", root.data.solution_x)

# check against binary solver in Gurobi
bin_model = Model(optimizer)
set_optimizer_attribute(bin_model, "OutputFlag", 0)
binary_model = build_unbounded_cardinality_model(bin_model,n,A,b)
x = binary_model[:x]
set_binary(x[bin] for bin in binary_vars)
optimize!(binary_model)
println("Exact solution: ", objective_value(binary_model) , " using ", value.(x))
println(" ")
println(" ")