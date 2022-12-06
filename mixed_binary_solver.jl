using StatsBase, Clarabel
include("tree.jl")
using JuMP, Gurobi, LinearAlgebra, Random, DataStructures, AbstractTrees

infinity = 1e3
""" this code is used to provide a mixed-binary-program solver where
the option of specifying which variables in x are binary is added to
the initial first_bnb_example
by default, branch_and_bound_solve_jump() has this list of binary variables as the whole set of variables"""

" only constrains the variables that are either binary or integer/natural to their interval
for relaxation, e.g. if binary: [0,1], if natural: [0,+Inf]
    the rest of the variables are completely free"
function add_constraints_jump(model::Model, lb, ub, integer_vars)
    x = model[:x]
    x = [x[i] for i in integer_vars]
    println("Adding constraints for vars: ", x)
    @constraint(model, lb, x.>= lb)
    if isnothing(ub)
        @constraint(model, ub, x.<= infinity)
    else
        @constraint(model, ub, x.<= ub)
    end
    print(model)
    return
end

"set upper bound to floor of fix_values if branch is left, or set lower bound to ceil of fix_value if branch is right "
function branch_variables_jump(x, fixed_x_indices, fix_values, bounds)
    if ~isnothing(fixed_x_indices) && ~isnothing(fix_values)
        for (i,j,k) in zip(fixed_x_indices, fix_values, bounds)
            if k == "ub"
                set_upper_bound(x[i],j)
            elseif k == "lb"
                set_lower_bound(x[i],j)
            end
        end
        
    end
    
end
function relax_variables_jump(x, integer_vars)
    # crucial step: relax any variables not in these vectors is currently fixed 
    # (because we changed to a different branch but model is same for all nodes)
    for i in integer_vars
        set_upper_bound(x[i],infinity)
        
        set_lower_bound(x[i],0.0)
    end
    
end

function build_unbounded_base_model(optimizer, n::Int,k::Int,Q::Matrix,c::Vector)
    model = Model()
    setOptimizer(model, optimizer)
    x = @variable(model, x[1:n])
    @objective(model, Min, x'*Q*x + c'*x)
    @constraint(model, sum_constraint, sum(x) == k)
    return model
end

function setOptimizer(model, optimizer)
    if optimizer == Gurobi.Optimizer
        set_optimizer(model, optimizer_with_attributes(optimizer, "OutputFlag" => 0))
    elseif optimizer == Clarabel.Optimizer
        set_optimizer(model,optimizer)
        set_optimizer_attribute(model, "verbose", false)
        # do this so that the solver doesn't scale the data
        set_optimizer_attribute(model, "equilibrate_enable", false)
    elseif optimizer == ECOS.Optimizer
        set_optimizer(model,optimizer)
        set_optimizer_attribute(model, "maxit", 100)
    end
    
end
" returns the upper bound computed when given the model as well as the rounded variable solution.
For Mixed_binary_solver: variables of indices from integer_vars (defaulted to all) are rounded based on relaxed_vars from lower_bound_model.
fixed_x_values is the vector of corresponding variables fixed on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub_jump(model::Model, optimizer, integer_vars, fixed_x_indices, fix_x_values, bounds, relaxed_vars)

    # set the relaxed variables equal to the rounded binary values
    # if these are in the set of binary variables    
    rounded_bounds = [round(value(relaxed_vars[i])) for i in integer_vars]
    # when model is a copy of another model, need to set the optimizer again
    setOptimizer(model, optimizer)
    println("rounded bounds vector: ", rounded_bounds)

    # con1 and con2 are of length rounded bounds/ integer_vars!
    con1 = model[:lb]
    con2 = model[:ub]
    x = model[:x]
    for i in 1:lastindex(rounded_bounds)
        set_normalized_rhs(con1[i] , rounded_bounds[i])
        set_normalized_rhs(con2[i], rounded_bounds[i])
    end
    
    # force the branching variables to fixed value
    branch_variables_jump(x, fixed_x_indices, fix_x_values,bounds)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        println("Ub computed with solution.x : ", value.(x))
        return objective_value(model), value.(x)
    else 
        println("Infeasible or unbounded problem for ub computation")
        return Inf, [Inf for _ in 1:length(relaxed_vars)]
    end
end
function printsolvervars(model, message)
    if optimizer == Clarabel.Optimizer
        solver = JuMP.unsafe_backend(model).solver
        println("SOLVER VARIABLES ",message," ", solver.variables)
        println("P in solver: ", solver.data.P)
        println("q in solver: ", solver.data.q)
        println("A in solver: ", solver.data.A)
        println("b in solver: ", solver.data.b)
        println("cones in solver: ", solver.cones.cone_specs)

    end
    
end
" return the lower bound as well as the values of x computed (for use by compute_ub_jump()).
model is given with relaxed constraints. fix_x_values is the vector of the
variables of fixed_x_indices that are currently fixed to a boolean"
function compute_lb_jump(model::Model, fixed_x_indices, fix_x_values, bounds, integer_vars)
    x = model[:x]
    #printsolvervars(model, "before lb")
    relax_variables_jump(x,integer_vars)
    branch_variables_jump(x,fixed_x_indices,fix_x_values, bounds)
    optimize!(model)
   # printsolvervars(model, "after lb")
    if termination_status(model) == MOI.OPTIMAL
        println("Values of relaxed solution ", value.(x))
        return objective_value(model), x
    else 
        println("Infeasible or unbounded problem for lb computation")
        println(model)

        return Inf, [Inf for _ in 1:length(x)]
    end
end

function termination_status_bnb(ub, lb,ϵ)
    if lb == Inf
        return "INFEASIBLE"
    elseif ub-lb <= ϵ
        return "OPTIMAL"
    end
    return "UNDEFINED" 
end

function solve_base_model(base_model::Model,integer_vars=collect(1:n))
    # natural variables relaxed to non-negative vars
    add_constraints_jump(base_model, zeros(length(integer_vars)), nothing, integer_vars) 
    optimize!(base_model)
    print("Solve_base_model: ",solution_summary(base_model))
end

"return the next variable to branch on/fix to binary value, splitting rule: most uncertain variable (i.e. closest to 0.5)
integer_vars is the SORTED list of binary variables within the model vars, only select from these
fixed_x_indices is the vector of already fixed variable indices, these should not be considered!"
function get_next_variable_to_fix_to_integer(x, integer_vars, fixed_x_indices)
    @assert issorted(integer_vars)
    remaining_branching_vars = setdiff(integer_vars, fixed_x_indices)
    if isempty(remaining_branching_vars)
        return -1
    end
    idx = remaining_branching_vars[1]
    for i in remaining_branching_vars # choose only from indices in integer_vars but not in fixed_x_indices!
        closest_int = floor(x[i])
        println("checking x...", i, " ", abs(x[i] -closest_int - 0.5))
        closest_int_idx = floor(x[idx])
        println("idx", idx, " ", abs(x[idx]- closest_int_idx - 0.5))

        if abs(x[i] -closest_int - 0.5) < abs(x[idx]- closest_int_idx - 0.5)
            idx = i 
        end
    end
    return idx
end
function branch_and_bound_solve_jump(base_model, optimizer, n, ϵ, integer_vars)

    if termination_status(base_model) == MOI.OPTIMAL
        lb =objective_value(base_model)
        println("Solution x of unbounded base model: ", value.(base_model[:x]))
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        ub, feasible_x=compute_ub_jump(copy(base_model), optimizer, integer_vars, nothing, nothing, nothing,value.(base_model[:x]))
        term_status = "UNDEFINED"
    else 
        term_status = "INFEASIBLE"
        error("Infeasible base problem")

    end
    # this is our root node of the binarytree
    root = BnbNode(MyNodeData(base_model,feasible_x,[],[],[],lb,ub))
    node = root

    # 3) start branching
    while term_status == "UNDEFINED"
        println("current node at depth ", node.data.depth, " has x as ", value.(node.data.model[:x]))
        #IMPORTANT: this x does NOT change after solving for l̃, ũ, l̄, ū
        # as value.() is performing broadcasting
        x = value.(node.data.model[:x]) 

        # which edge to split along i.e. which variable to fix next?
        fixed_x_index = get_next_variable_to_fix_to_integer(value.(x), integer_vars, node.data.fixed_x_ind) 
        println("GOT BRANCHING VARIABLE: ", fixed_x_index, " SET SMALLER THAN FLOOR (left): ", floor(x[fixed_x_index]), " OR GREATER THAN CEIL (right)", ceil(x[fixed_x_index]))
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always sets lower bound of the branching variable to closest lower integer
        fixed_x_left = vcat(node.data.fixed_x_values, floor(x[fixed_x_index])) 
        bounds_left = vcat(node.data.bounds, "ub")
        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_model = node.data.model 

        l̃, relaxed_x_left = compute_lb_jump(left_model, fixed_x_indices, fixed_x_left, bounds_left,integer_vars)
        println("solved for l̃: ", l̃)

        ũ, feasible_x_left = compute_ub_jump(copy(left_model), optimizer,integer_vars,fixed_x_indices, fixed_x_left, bounds_left, relaxed_x_left)
        println("solved for ũ: ", ũ)

        println("branched vars on left branch are : ", fixed_x_indices, " to ", fixed_x_left, " with bounds ", bounds_left)
        
        #create new child node (left)
        
        left_node = leftchild!(node, MyNodeData(left_model, feasible_x_left, fixed_x_indices, fixed_x_left,bounds_left, l̃,ũ))

        # solve the right child problem to get l-bar and u-bar
        right_model = node.data.model 
        fixed_x_right = vcat(node.data.fixed_x_values, ceil(x[fixed_x_index])) 
        bounds_right = vcat(node.data.bounds, "lb")
        l̄, relaxed_x_right = compute_lb_jump(right_model, fixed_x_indices, fixed_x_right, bounds_right, integer_vars)
        
        println("solved for l̄: ", l̄)
        ū, feasible_x_right = compute_ub_jump(copy(right_model), optimizer, integer_vars, fixed_x_indices, fixed_x_right, bounds_right,relaxed_x_right)
        println("solved for ū: ", ū)

        println("branched vars on right branch are : ", fixed_x_indices, " to ", fixed_x_right, " with bounds ", bounds_right)
        #create new child node (right)
        right_node = rightchild!(node, MyNodeData(right_model, feasible_x_right, fixed_x_indices,fixed_x_right, bounds_right,l̄,ū))

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
        compute_lb_jump(node.data.model, node.data.fixed_x_ind, node.data.fixed_x_values,node.data.bounds,integer_vars)

        println("Difference: ", root.data.ub, " - ",root.data.lb, " is ",root.data.ub-root.data.lb )
        println(" ")
        term_status = termination_status_bnb(root.data.ub,root.data.lb, ϵ)

    end
    return root, term_status
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
integer_vars = sample(1:n, m, replace = false)
sort!(integer_vars)
root = branch_and_bound_solve_jump(base_model,optimizer,n,ϵ, integer_vars)
println("Found objective: ", root.data.ub, " using ", root.data.solution_x)

include("brute_recursion.jl")
# check against binary solver in Gurobi
bin_model = Model(optimizer)
set_optimizer_attribute(bin_model, "OutputFlag", 0)
binary_model = build_base_model(bin_model,n,k,Q,c,integer_vars)
optimize!(binary_model)
println("Exact solution: ", objective_value(binary_model) , " using ", value.(binary_model[:x])) =#