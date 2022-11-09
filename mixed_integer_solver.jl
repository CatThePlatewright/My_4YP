include("mixed_binary_solver.jl")
# imported functions from mixed_binary_solver:
# add_constraints, fix_variables(), build_unbounded_base_model()


" returns the upper bound computed when given the model as well as the feasible variable solution.
variables of indices from integer_vars (defaulted to all) are rounded based on relaxed_vars from lower_bound_model.
fixed_x_values is the vector of corresponding variables fixed on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub(model::Model, optimizer, integer_vars,fixed_x_indices, fix_x_values, relaxed_vars)

    # set the relaxed variables equal to the rounded binary values
    # if these are in the set of binary variables    
    rounded_bounds = [round(value(relaxed_vars[i])) for i in integer_vars]
    # when model is a copy of another model, need to set the optimizer again
    set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    println("rounded bounds vector: ", rounded_bounds)

    con1 = model[:lb_constraint]
    con2 = model[:ub_constraint]
    x = model[:x]
    for (i, j) in zip(integer_vars,1:length(rounded_bounds))
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
variables of fixed_x_indices that are currently fixed to an integer"
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
integer_vars is the SORTED list of binary variables within the model vars, only select from these"
function get_next_variable_to_fix_to_integer(x, integer_vars)
    @assert issorted(integer_vars)
    idx = integer_vars[1]
    for i in integer_vars
        closest_int = round(x[i])
        closest_int_idx = round(x[idx])
        if abs(x[i] -closest_int - 0.5) < abs(x[idx]- closest_int_idx - 0.5)
            idx = i 
        end
    end
    return idx
end


function branch_and_bound_solve(base_model, optimizer, n, ϵ, integer_vars=collect(1:n))
    # natural variables relaxed to non-negative vars
    add_constraints(base_model, zeros(n), nothing) 
    optimize!(base_model)

    if termination_status(base_model) == MOI.OPTIMAL
        lb =objective_value(base_model)
        println("Solution x of unbounded base model: ", value.(base_model[:x]))
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        ub, feasible_x=compute_ub(copy(base_model), optimizer, integer_vars, nothing, nothing, value.(base_model[:x]))
    else 
        println("Infeasible or unbounded problem ")
    end
    # this is our root node of the binarytree
    #TODO change node name
    root = BinaryNode(MyNodeData(base_model,feasible_x,[],[],lb,ub))
    node = root

    # 3) start branching
    while (root.data.ub-root.data.lb > ϵ) && (node.data.depth < 20)
        println("current node at depth ", node.data.depth, " has x as ", value.(node.data.model[:x]))
        x = value.(node.data.model[:x])
        # which edge to split along i.e. which variable to fix next?
        fixed_x_index = get_next_variable_to_fix_to_integer(value.(x), integer_vars) 
        println("GOT NEXT VAR TO FIX: ", fixed_x_index, " TO FLOOR : ", floor(x[fixed_x_index]), " TO CEIL ", ceil(x[fixed_x_index]))
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always fixes the next variable to closest lower integer
        fixed_x_left = vcat(node.data.fixed_x_values, floor(x[fixed_x_index])) 
        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_model = node.data.model 
        l̃, relaxed_x_left = compute_lb(left_model, fixed_x_indices, fixed_x_left)
        println("solved for l̃: ", l̃)
        ũ, feasible_x_left = compute_ub(copy(left_model), optimizer,integer_vars,fixed_x_indices, fixed_x_left, relaxed_x_left)
        println("solved for ũ: ", ũ)
        println("fixed indices on left branch are : ", fixed_x_indices, " to ", fixed_x_left)
        
        #create new child node (left)
        
        left_node = leftchild!(node, MyNodeData(left_model, feasible_x_left, fixed_x_indices, fixed_x_left,l̃,ũ))

        # solve the right child problem to get l-bar and u-bar
        right_model = node.data.model
        fixed_x_right = vcat(node.data.fixed_x_values, ceil(x[fixed_x_index])) 
        l̄, relaxed_x_right = compute_lb(right_model, fixed_x_indices, fixed_x_right)
        
        println("solved for l̄: ", l̄)
        ū, feasible_x_right = compute_ub(copy(right_model), optimizer, integer_vars, fixed_x_indices, fixed_x_right, relaxed_x_right)
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

optimizer = Gurobi.Optimizer
n = 5
k= 4
m = 2 # how many integer variables (if mixed integer problem)
Q = Matrix{FloatT}(I, n, n) 
Random.seed!(1234)
c = rand(FloatT,n)
ϵ = 0.00000001

base_model = build_unbounded_base_model(optimizer,n,k,Q,c)
integer_vars = sample(1:n, m, replace = false)
sort!(integer_vars)
root = branch_and_bound_solve(base_model,optimizer,n,ϵ, integer_vars)
println("Found objective: ", root.data.ub, " using ", root.data.solution_x)

# check against binary solver in Gurobi
bin_model = Model(optimizer)
set_optimizer_attribute(bin_model, "OutputFlag", 0)
x = @variable(bin_model, x[i = 1:n] >= 0.0)
for bin in integer_vars
    set_integer(x[bin])
end
@objective(bin_model, Min, x'*Q*x + c'*x)
@constraint(bin_model, sum_constraint, sum(x) == k)
optimize!(bin_model)
println("Exact solution: ", objective_value(bin_model) , " using ", value.(bin_model[:x])) 