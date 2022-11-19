using Clarabel, JuMP

model = JuMP.Model(Clarabel.Optimizer)
set_optimizer_attribute(model, "verbose", false)

# do this so that the solver doesn't scale the data
set_optimizer_attribute(model, "equilibrate_enable", false)
n = 4
k= 2
m = 3 # how many integer variables (if mixed integer problem)
Q = Matrix{Float64}(I, n, n) 
Random.seed!(1234)
c = rand(Float64,n)
ϵ = 0.00000001

# example problem (details not important)
@variable(model, x[1:n])
@objective(model, Min, x'*Q*x + c'*x)
@constraint(model, sum_constraint, sum(x) == k)
@constraint(model, c, 0 .<= x .<= 1)
optimize!(model)
print(model)
# access the Clarabel solver object
solver = JuMP.unsafe_backend(model).solver

# now you can get data etc
data = solver.data
println(data)
P = data.P
q = data.q
A = data.A
b = data.b
s = solver.cones
