#=  Changes to brute_recursion.jl:
1. Rebuilds the model each time, instead of copying it
2. Vectorised constraint version instead of variable bound. 
Less constraints overall since relaxed constraints are replaced by fix-variable constraint.

=#
include("tree.jl")
using JuMP, Gurobi, LinearAlgebra, Random, DataStructures, AbstractTrees

function add_constraints(model::Model, lb, ub)
    x = model[:x]
    @constraint(model, lb_constraint, x .>= lb)
    @constraint(model, ub_constraint, x .<= ub)

    return
end

function fix_variable(model::Model, i::Int, value::Float64)
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


" returns the upper bound computed when given the model as well as the rounded variable solution.
all variables are rounded based on relaxed_vars from lower_bound_model.
fixed_x is the variable you fix on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub(model::Model, optimizer, fixed_x_index, fix_value, relaxed_vars)

    # set the relaxed variables equal to the rounded binary values
    # need to update the rhs of the vectorised ub and lb constraints
    rounded_bounds = round.(value.(relaxed_vars))
    println("Rounded vars:", rounded_bounds)

    # when model is a copy of another model, need to set the optimizer again
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))

    con1 = model[:lb_constraint]
    con2 = model[:ub_constraint]
    x = model[:x]

    for i in 1:length(relaxed_vars)
        set_normalized_rhs(con1[i] , rounded_bounds[i])
        set_normalized_rhs(con2[i], rounded_bounds[i])
    end
    # force the branching variables to fixed value
    if ~isnothing(fixed_x_index) && ~isnothing(fix_value)
        fix(x[fixed_x_index], fix_value; force = true)
    end
    
    optimize!(model)
    #TODO: write feasibility check function for this if-else
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x)
    else 
        println("Infeasible or unbounded problem for ub computation")
        return Inf, [Inf for _ in 1:length(relaxed_vars)]
    end
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fixed_x is the
branching variable to be set to fixed value"
function compute_lb(model::Model, fixed_x_index, fix_value::Float64)
    x = model[:x]
    fix(x[fixed_x_index], fix_value; force = true) # overrides previous relaxed constraint
    optimize!(model)
    #TODO: write feasibility check function for this if-else
    if termination_status(model) == MOI.OPTIMAL
        println("Values of relaxed solution ", value.(x))
        return objective_value(model), x
    else 
        println("Infeasible or unbounded problem for lb computation")
        return Inf, [Inf for _ in 1:length(x)]
    end
end

"return the next variable to branch on/fix to binary value, splitting rule: most uncertain variable (i.e. closest to 0.5)"
function get_next_variable_to_fix(x_values::Vector{Float64})
    return argmin(abs.(x_values .- 0.5))
end
# model parameters
optimizer = Gurobi.Optimizer
n = 5
k = 3
Q = Matrix{Float64}(I, n, n) 
Random.seed!(1234)
c = rand(Float64,n)

# build the root node
# 1) compute L1, lower bound on p* of mixed Boolean problem (p.5 of BnB paper)

base_model = build_unbounded_base_model(optimizer,n,k,Q,c)
add_constraints(base_model, zeros(n), ones(n)) # binary case would make ub and lb constraints redundant!
optimize!(base_model)

if termination_status(base_model) == MOI.OPTIMAL
    lb =objective_value(base_model)
    # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
    # IMPORTANT: argument must be copy(model) to avoid overwriting relaxed constraint!
    ub, feasible_x=compute_ub(copy(base_model), optimizer, nothing, nothing, value.(base_model[:x]))
else 
    println("Infeasible or unbounded problem ")
end
# this is our root node of the binarytree
root = BinaryNode(MyNodeData(base_model,feasible_x,[],[],lb,ub))
node = root
ϵ = 0.00000001

# 3) start branching
while (root.data.ub-root.data.lb > ϵ) && node.data.depth < 3 #&& ~isinf(root.data.ub-root.data.lb)
    global node
    println("current node has ", value.(node.data.model[:x]))
    local x = value.(node.data.model[:x])
    # which edge to split along i.e. which variable to fix next?
    fixed_x_index = get_next_variable_to_fix(value.(x)) 
    println("got next variable to fix: ", fixed_x_index)
    # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
    left_model = node.data.model 
    l̃, relaxed_x_left = compute_lb(left_model, fixed_x_index, 0.0)
    println("solved for l̃: ", l̃)
    ũ, feasible_x_left = compute_ub(copy(left_model), optimizer,fixed_x_index, 0.0, relaxed_x_left)
    println("solved for ũ: ", ũ)
    
    #create new child node (left)
    fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
    fixed_xs = vcat(node.data.fixed_x_values,0.0)
    left_node = leftchild!(node, MyNodeData(left_model, feasible_x_left, fixed_x_indices, fixed_xs,l̃,ũ))

    # solve the right child problem to get l-bar and u-bar
    right_model = node.data.model
    l̄, relaxed_x_right = compute_lb(right_model, fixed_x_index, 1.0)
    
    println("solved for l̄: ", l̄)
    ū, feasible_x_right = compute_ub(copy(right_model), optimizer, fixed_x_index, 1.0, relaxed_x_right)
    println("solved for ū: ", ū)
    fixed_xs = vcat(node.data.fixed_x_values,1.0)
    println("fixed indices are : ", fixed_x_indices, " to ", fixed_xs)
    #create new child node (right)
    right_node = rightchild!(node, MyNodeData(right_model, feasible_x_right, fixed_x_indices,fixed_xs,l̄,ū))

    # new lower and upper bounds on p*: pick the minimum
    λ =  minimum([l̄, l̃]) 
    μ =  minimum([ū, ũ])
    # TODO: not if both are Inf? need to trace back and follow other path!

    node_with_best_lb = λ==l̄ ? right_node : left_node
    node_with_best_ub = μ==ū ? right_node : left_node
    #back-propagate the new lb and ub to root: TOASK: BUT NOT FOR i = 1!
    update_best_ub(node_with_best_ub)
    update_best_lb(node_with_best_lb)

    println("root bounds: ", root.data.lb,"    ", root.data.ub)

    # decide which child node to branch on next: pick the one with the lowest lb
    node = branch_from_node(root) #start from root at every iteration, trace down to the max. depth
    println("Difference: ", root.data.ub, " - ",root.data.lb, " is ",root.data.ub-root.data.lb )
    println(" ")
end

#print_tree(root)

include("brute_recursion.jl")
# having ub-lb < ϵ, test-solve for the problem again using incumbent solution_x 
final_solution = root.data.solution_x
println("Final solution: ", root.data.ub, " using: ", final_solution)

bin_model = Model(optimizer)
set_optimizer_attribute(bin_model, "OutputFlag", 0)

binary_model = build_base_model(bin_model,n,k,Q,c,true)
optimize!(binary_model)

println("Binary exact solution: ", objective_value(binary_model), " using: ", value.(binary_model[:x]))