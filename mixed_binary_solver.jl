using StatsBase
include("tree.jl")
using JuMP, Gurobi, LinearAlgebra, Random, DataStructures, AbstractTrees

""" this code is used to provide a mixed-binary-program solver where
the option of specifying which variables in x are binary is added to
the initial first_bnb_example
by default, branch_and_bound_solve() has this list of binary variables as the whole set of variables"""
function add_constraints(model::Model, lb, ub)
    x = model[:x]
    @constraint(model, lb_constraint, x .>= lb)
    @constraint(model, ub_constraint, x .<= ub)

    return
end

function fix_variables(x, fixed_x_indices, fix_values)
    # force these variables to be fixed
    if ~isnothing(fixed_x_indices) && ~isnothing(fix_values)
        for index in fixed_x_indices, j in 1:length(fixed_x_indices)
            fix(x[index], fix_values[j]; force = true)
        end
        # crucial step: relax any variables not in these vectors is currently fixed 
        # (because we changed to a different branch but model is same for all nodes)
        for i in 1:length(x)
            if ~(i in fixed_x_indices) && is_fixed(x[i])
                unfix(x[i])
            end
        end
    end
    
end


function build_unbounded_base_model(optimizer, n::Int,k::Int,Q::Matrix,c::Vector)
    model = Model()
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    x = @variable(model, x[1:n])
    @objective(model, Min, x'*Q*x + c'*x)
    @constraint(model, sum_constraint, sum(x) == k)
    return model
end


" returns the upper bound computed when given the model as well as the rounded variable solution.
For Mixed_binary_solver: variables of indices from binary_vars (defaulted to all) are rounded based on relaxed_vars from lower_bound_model.
fixed_x_values is the vector of corresponding variables fixed on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub(model::Model, optimizer, binary_vars,fixed_x_indices, fix_x_values, relaxed_vars)

    # set the relaxed variables equal to the rounded binary values
    # if these are in the set of binary variables    
    rounded_bounds = [round(value(relaxed_vars[i])) for i in binary_vars]
    # when model is a copy of another model, need to set the optimizer again
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    println("rounded bounds vector: ", rounded_bounds)

    con1 = model[:lb_constraint]
    con2 = model[:ub_constraint]
    x = model[:x]
    for (i, j) in zip(binary_vars,1:length(rounded_bounds))
        set_normalized_rhs(con1[i] , rounded_bounds[j])
        set_normalized_rhs(con2[i], rounded_bounds[j])
    end
    # force the branching variables to fixed value
    fix_variables(x, fixed_x_indices, fix_x_values)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return objective_value(model), value.(x)
    else 
        println("Infeasible or unbounded problem for ub computation")
        return Inf, [Inf for _ in 1:length(relaxed_vars)]
    end
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fix_x_values is the vector of the
variables of fixed_x_indices that are currently fixed to a boolean"
function compute_lb(model::Model, fixed_x_indices, fix_x_values)
    x = model[:x]
    fix_variables(x,fixed_x_indices,fix_x_values)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        println("Values of relaxed solution ", value.(x))
        return objective_value(model), x
    else 
        println("Infeasible or unbounded problem for lb computation")
        return Inf, [Inf for _ in 1:length(x)]
    end
end

"return the next variable to branch on/fix to binary value, splitting rule: most uncertain variable (i.e. closest to 0.5)
Binary_vars is the SORTED list of binary variables within the model vars, only select from these
fixed_vars is the vector of already fixed variable indices, these should not be considered!"
function get_next_variable_to_fix(x, binary_vars, fixed_vars)
    @assert issorted(binary_vars)
    idx = setdiff(binary_vars,fixed_vars)[1]
    for i in setdiff(binary_vars,fixed_vars)
        if abs(x[i] - 0.5) < abs(x[idx]-0.5)
            idx = i 
        end
    end
    return idx
end


function branch_and_bound_solve(base_model, optimizer, n, ϵ, binary_vars=collect(1:n))
    
    # 1) compute L1, lower bound on p* of mixed Boolean problem (p.5 of BnB paper)
    add_constraints(base_model, zeros(n), ones(n)) # binary case would make ub and lb constraints redundant!
    optimize!(base_model)

    if termination_status(base_model) == MOI.OPTIMAL
        lb =objective_value(base_model)
        println("Values base model: ", value.(base_model[:x]))
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        # IMPORTANT: argument must be copy(model) to avoid overwriting relaxed constraint!
        ub, feasible_x=compute_ub(copy(base_model), optimizer, binary_vars, nothing, nothing, value.(base_model[:x]))
    else 
        println("Infeasible or unbounded problem ")
    end
    # this is our root node of the binarytree
    root = BinaryNode(MyNodeData(base_model,feasible_x,[],[],lb,ub))
    node = root

    # 3) start branching
    while (root.data.ub-root.data.lb > ϵ) 
        println("current node at depth ", node.data.depth, " has x as ", value.(node.data.model[:x]))
        x = value.(node.data.model[:x])
        # which edge to split along i.e. which variable to fix next?
        # for Mixed_binary_solver: only select among vars specified as binary
        fixed_x_index = get_next_variable_to_fix(value.(x), binary_vars, node.data.fixed_x_ind) 
        println("GOT NEXT VAR TO FIX: ", fixed_x_index)
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always fixes the next variable to 0
        fixed_x_left = vcat(node.data.fixed_x_values,0.0) 
        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_model = node.data.model 
        l̃, relaxed_x_left = compute_lb(left_model, fixed_x_indices, fixed_x_left)
        println("solved for l̃: ", l̃)
        ũ, feasible_x_left = compute_ub(copy(left_model), optimizer,binary_vars,fixed_x_indices, fixed_x_left, relaxed_x_left)
        println("solved for ũ: ", ũ)
        
        #create new child node (left)
        
        left_node = leftchild!(node, MyNodeData(left_model, feasible_x_left, fixed_x_indices, fixed_x_left,l̃,ũ))

        # solve the right child problem to get l-bar and u-bar
        right_model = node.data.model
        # left branch always fixes the next variable to 0
        fixed_x_right = vcat(node.data.fixed_x_values,1.0) 
        l̄, relaxed_x_right = compute_lb(right_model, fixed_x_indices, fixed_x_right)
        
        println("solved for l̄: ", l̄)
        ū, feasible_x_right = compute_ub(copy(right_model), optimizer, binary_vars, fixed_x_indices, fixed_x_right, relaxed_x_right)
        println("solved for ū: ", ū)
        println("fixed indices on right branch are : ", fixed_x_indices, " to ", fixed_x_right)
        #create new child node (right)
        right_node = rightchild!(node, MyNodeData(right_model, feasible_x_right, fixed_x_indices,fixed_x_right,l̄,ū))

        # new lower and upper bounds on p*: pick the minimum
        λ =  minimum([l̄, l̃]) 
        μ =  minimum([ū, ũ])

        node_with_best_lb = λ==l̄ ? right_node : left_node
        node_with_best_ub = μ==ū ? right_node : left_node
        #back-propagate the new lb and ub to root: TOASK: BUT NOT FOR i = 1!
        update_best_ub(node_with_best_ub)
        update_best_lb(node_with_best_lb)

        println("root bounds: ", root.data.lb,"    ", root.data.ub)

        # decide which child node to branch on next: pick the one with the lowest lb
        node = branch_from_node(root) #start from root at every iteration, trace down to the max. depth
        update_best_lb(node)
        # CRUCIAL: solve again to get the correct x vector but feasible x solution stored in data.solution_x
        compute_lb(node.data.model, node.data.fixed_x_ind, node.data.fixed_x_values)

        println("Difference: ", root.data.ub, " - ",root.data.lb, " is ",root.data.ub-root.data.lb )
        println(" ")
    end
    return root
end

#= optimizer = Gurobi.Optimizer
n = 8
k= 5
m = 4 # how many binary variables
Q = Matrix{FloatT}(I, n, n) 
Random.seed!(1234)
c = rand(FloatT,n)
ϵ = 0.00000001

base_model = build_unbounded_base_model(optimizer,n,k,Q,c)
binary_vars = sample(1:n, m, replace = false)
sort!(binary_vars)
root = branch_and_bound_solve(base_model,optimizer,n,ϵ, binary_vars)
println("Found objective: ", root.data.ub, " using ", root.data.solution_x)

include("brute_recursion.jl")
# check against binary solver in Gurobi
bin_model = Model(optimizer)
set_optimizer_attribute(bin_model, "OutputFlag", 0)
binary_model = build_base_model(bin_model,n,k,Q,c,binary_vars)
optimize!(binary_model)
println("Exact solution: ", objective_value(binary_model) , " using ", value.(binary_model[:x])) =#