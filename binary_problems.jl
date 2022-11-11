include("mixed_binary_solver.jl")
FloatT = Float64
optimizer = Gurobi.Optimizer
n = 2
m = 2 # how many binary variables
Random.seed!(1234)
# create orthogonal V matrix
V = rand(n,n)
V= qr(V).Q 

# create N matrices stored in A 
A  = Matrix{FloatT}[]
for _ in 1:n
    local a = rand(-10:10,n) # TOASK does it need to be n or just some smaller dimension?
    push!(A, V*diagm(a)*V') # these are the individual A matrices
end
ϵ = 0.00001
function build_unbounded_cardinality_model(optimizer, n::Int,A::Vector)
    model = Model()
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    @variable(model, x[1:n])
    @objective(model, Min, sum(x)) # emulating the cardinality of x
    c = sum(A[i]*x[i] for  i = 1:n)
    @constraint(model, con, c .>= 0) 
    return model
end
#= base_model = build_unbounded_cardinality_model(optimizer,n,A)
binary_vars = sample(1:n, m, replace = false)
sort!(binary_vars)
root = branch_and_bound_solve(base_model,optimizer,n,ϵ, binary_vars)
@test termination_status(root.data.model)   == OPTIMAL
println("Found objective: ", root.data.ub, " using ", root.data.solution_x) =#

# check against binary solver in Gurobi
binary_model = build_unbounded_cardinality_model(optimizer,n,A)
binary_vars = sample(1:n, m, replace = false)

x = binary_model[:x]
optimize!(binary_model)
println("Exact solution: ", objective_value(binary_model) , " using ", value.(x))
