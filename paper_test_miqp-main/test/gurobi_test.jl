#Gurobi for test
using Gurobi, JuMP, LinearAlgebra

q = [1; 1.; 2.];
P = [2. -1. -1.; -1. 3. 1.; -1. 1. 2.];
A = -[2. 1. 1.];
l = -[-1.];
u = [3.];
m, n = size(A);
li = -3*ones(n);	#relaxed box constraints for integer constraints
ui = 3*ones(n);		#use datatype Float64 for simplicity
b = zeros(m);

# m = 5
# n = 10
# A = randn(m,n)
# b = randn(m)
# M = 1
# k = convert(Int64, round(m/3))
#
# S = A'*A
# c = -2*A'*b
# d = norm(b)^2

# Define the model
# ----------------

model = Model(with_optimizer(Gurobi.Optimizer)) # define name of the model, it could be anything, not necessarily "model"

# Variables
# ---------

@variable(model, x[1:3], Int) # define variable x

# Objective
# ---------

sense = MOI.MIN_SENSE # by this command, we are programatically defining a quadratic objective to be minimized

@objective(model, sense, 1/2 * x' * P * x + q' * x) # define the objective

# Constraints
# ------------

@constraint(model, box_cons, l .<= b - A*x .<= u) # lower bound constraint

@constraint(model, var_cons, li .<= x .<= ui) # upper bound constraint

# Run the optimizer
# -----------------

status=optimize!(model) # time to optimize!

# Let us look at the important outputs
# ------------------------------------
println("******************************************************")
println("optimal objective value is = ", objective_value(model))
println("optimal x is = ",  value.(x))
