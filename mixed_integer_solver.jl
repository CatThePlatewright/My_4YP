include("mixed_binary_solver.jl")
using Test, ECOS
# imported functions from mixed_binary_solver:
# add_constraints, fix_variables(), build_unbounded_base_model()


#=     =#

"return the next variable to branch on/fix to binary value, splitting rule: most uncertain variable (i.e. closest to 0.5)
integer_vars is the SORTED list of binary variables within the model vars, only select from these
fixed_x_indices is the vector of already fixed variable indices, these should not be considered!"
function get_next_variable_to_fix_to_integer(x, integer_vars, fixed_x_indices)
    @assert issorted(integer_vars)
    idx = setdiff(integer_vars, fixed_x_indices)[1]
    for i in setdiff(integer_vars, fixed_x_indices) # choose only from indices in integer_vars but not in fixed_x_indices!
        closest_int = round(x[i])
        closest_int_idx = round(x[idx])
        if abs(x[i] -closest_int - 0.5) < abs(x[idx]- closest_int_idx - 0.5)
            idx = i 
        end
    end
    return idx
end


function branch_and_bound_solve(base_model, optimizer, n, ϵ, integer_vars=collect(1:n))

    if termination_status(base_model) == MOI.OPTIMAL
        lb =objective_value(base_model)
        println("Solution x of unbounded base model: ", value.(base_model[:x]))
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        ub, feasible_x=compute_ub(copy(base_model), optimizer, integer_vars, nothing, nothing, value.(base_model[:x]))
        term_status = "UNDEFINED"
    else 
        term_status = "INFEASIBLE"
        error("Infeasible base problem")

    end
    # this is our root node of the binarytree
    root = BinaryNode(MyNodeData(base_model,feasible_x,[],[],lb,ub))
    node = root

    # 3) start branching
    while term_status == "UNDEFINED"
        println("current node at depth ", node.data.depth, " has x as ", value.(node.data.model[:x]))
        x = value.(node.data.model[:x])
        # which edge to split along i.e. which variable to fix next?
        fixed_x_index = get_next_variable_to_fix_to_integer(value.(x), integer_vars, node.data.fixed_x_ind) 
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
        term_status = termination_status_bnb(root.data.ub,root.data.lb, ϵ)

    end
    return root, term_status
end
function solve_base_model(base_model::Model,integer_vars)
    # natural variables relaxed to non-negative vars
    add_constraints(base_model, zeros(length(integer_vars)), nothing, integer_vars) 
    optimize!(base_model)
    print(solution_summary(base_model))
end

optimizer = Clarabel.Optimizer
n = 2
k= 1
m = 2 # how many integer variables (if mixed integer problem)
integer_vars = sample(1:n, m, replace = false)
sort!(integer_vars)
println("Integer variables : ", integer_vars)
Q = Matrix{Float64}(I, n, n) 
Random.seed!(1234)
c = rand(Float64,n)
ϵ = 0.00000001

function main()
    

    # check against binary solver in Gurobi
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)
    x = @variable(exact_model, x[i = 1:n])
    for bin in integer_vars
        set_integer(x[bin])
    end
    @objective(exact_model, Min, x'*Q*x + c'*x)
    @constraint(exact_model, sum_constraint, sum(x) == k)
    optimize!(exact_model)
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 


    
    base_model = build_unbounded_base_model(optimizer,n,k,Q,c)

    solve_base_model(base_model,integer_vars)

    root, term_status = branch_and_bound_solve(base_model,optimizer,n,ϵ, integer_vars)
    @test term_status == "OPTIMAL"
    println("Found objective: ", root.data.ub, " using ", root.data.solution_x)


    tol = 1e-4

    @test isapprox(norm(root.data.solution_x - Float64.(value.(exact_model[:x]))), zero(Float64), atol=tol)
    @test isapprox(root.data.ub, Float64(objective_value(exact_model)), atol=tol)
        
end