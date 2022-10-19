#=  Changes to brute_recursion.jl:
1. Rebuilds the model each time, instead of copying it
2. Vectorised constraint version instead of variable bound. 
Less constraints overall since relaxed constraints are replaced by fix-variable constraint.

=#
using JuMP, Gurobi, LinearAlgebra, Random, DataStructures

function add_constraints(model::Model, lb, ub)
    x = model[:x]
    @constraint(model, lb_constraint, x .>= lb)
    @constraint(model, ub_constraint, x .<= ub)

    return
end

function fix_variable(model::Model, i::Int, value::Float16)
    x = model[:x]
    con1 = model[:lb_constraint]
    con2 = model[:ub_constraint]
    set_normalized_rhs(con1[i], value)
    set_normalized_rhs(con2[i], value)
end

function add_variables(
    model::Model,n::Int, binary)
    if binary
        return @variable(model, x[1:n], Bin)
    end
    return @variable(model, x[1:n]) #free variables, then add constraint vector
end

function build_unbounded_base_model(optimizer, n::Int,k::Int,Q::Matrix,c::Vector, binary = false)
    model = Model()
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    x = add_variables(model, n, binary)
    @objective(model, Min, x'*Q*x + c'*x)
    @constraint(model, sum_constraint, sum(x) == k)
    return model
end

function build_child_model(model::Model, i::Int, fix_to_value, final_p_values)
    fix_variable(model, i, Float16(fix_to_value))
        # Or: new_model, reference_map = copy_model(model)
        #  x_new = reference_map[x]
    x = model[:x]
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        if i == n
            println(model)
            println("Optimal values for x:", value.(x))
            println("Optimal Objective", objective_value(model))
            push!(final_p_values, objective_value(model))
        end
        return objective_value(model)
    else
        return Inf
    end
    
end

" returns the upper bound computed when given the model.
all variables are rounded based on relaxed_vars from lower_bound_model.
fixed_x is the variable you fix on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub(model::Model, fixed_x_index, value, relaxed_vars)

    # set the relaxed variables equal to the rounded binary values
    # need to update the rhs of the vectorised ub and lb constraints
    rounded_bounds = round.(value.(relaxed_vars))
    con1 = model[:lb_constraint]
    con2 = model[:ub_constraint]
    set_normalized_rhs(con1[i], rounded_bounds[i] for i in 1:n)
    set_normalized_rhs(con2[i], rounded_bounds[i] for i in 1:n)

    # force the branching variables to fixed value
    if ~isnothing(fixed_x_index) && ~isnothing(value)
        x = model[:x]
        fix(x[fixed_x_index], value; force = true)
    end

    optimize!(model)
    #TODO: write feasibility check function for this if-else
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    else 
        println("Infeasible or unbounded problem")
        return Inf
        #TODO: terminate the node here
    end
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fixed_x is the
branching variable to be set to fixed value"
function compute_lb(model::Model, fixed_x_index, value::Float16)
    x = model[:x]
    fix(x[fixed_x_index], value; force = true) # overrides previous relaxed constraint
    optimize!(model)
    #TODO: write feasibility check function for this if-else
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), x
    else 
        println("Infeasible or unbounded problem")
        return Inf
        #terminate the node here
    end
end

# model parameters
optimizer = Gurobi.Optimizer
n = 4
k = 2
Q = Matrix{Float16}(I, n, n) 
Random.seed!(1234)
c = rand(Float16,n)
ub = Vector{Float64}() 
lb = Vector{Float64}() 
final_p_values = Vector{Float64}()

# 1) compute L1, lower bound on p* of mixed Boolean problem (p.5 of BnB paper)
base_model = build_unbounded_base_model(optimizer,n,k,Q,c)
x = base_model[:x]
add_constraints(base_model, zeros(n), ones(n)) # binary case would make ub and lb constraints redundant!
optimize!(base_model)

if termination_status(model) == MOI.OPTIMAL
    push!(lb,objective_value(base_model))
    # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
    push!(ub, compute_ub(base_model, nothing, nothing, x))
else 
    error("Infeasible or unbounded problem")
    # TODO: terminate
end


# 3) start branching, for now, just stupidly by order of variable index
i = 1
eps = 0.00001
model_queue = Queue{Model}() # queue to store current model for later copying
enqueue!(model_queue, base_model)

while (ub[i]-lb[i] > eps) 
    # TODO: maybe  for all model in Queue, dequeue?
    # so the index i will not go out of bounds 2 lines down?
    while (~isempty(Queue))
        parent_model = dequeue!(Queue)
        left_model = copy(parent_model)

        l_wiggle, relaxed_x_left = compute_lb(left_model, i, 0.0)
        u_wiggle = compute_ub(left_model, i, 0.0, relaxed_x_left)

        right_model = copy(parent_model)
        l_bar, relaxed_x_right = compute_lb(right_model, i, 1.0)
        u_bar = compute_ub(right_model, i, 0.0, relaxed_x_right)
    end
    # new lower and upper bounds on p*
    # TODO: this should take minimum over all l_bars etc???
    lb[i+1] =  minimum([l_bar, l_wiggle])
    ub[i+1] =  minimum([u_bar, u_wiggle])

    i += 1

    enqueue!(Queue, left_model, right_model)
    #TODO: figure out the order of dequeueing
end