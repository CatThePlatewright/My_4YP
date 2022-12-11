using FileIO, JLD, JLD2 # to load the matrices from the file
using JuMP, Gurobi

#import data
P = load("problem.jld","P")
q = load("problem.jld","q")
A = load("problem.jld","A")
b = load("problem.jld","b")
l = load("problem.jld","l")
u = load("problem.jld","u")
lb = load("problem.jld","lb")
ub = load("problem.jld","ub")
i_idx = load("problem.jld","i_idx")
m = length(b)
n = length(lb)

model = Model(Gurobi.Optimizer)
@variable(model, x[1:n]);
set_integer.(x[i_idx])  #set integer constraints
@objective(model, Min, 0.5*x'*P*x + q' * x )
# @constraint(model, Symmetric(B - A1  .* x[1] - A2 .* x[2] )  in JuMP.PSDCone());
@constraints(model, begin
    x .>= lb
    x .<= ub
    b - A*x .>= l
    b - A*x .<= u
end)

optimize!(model)
