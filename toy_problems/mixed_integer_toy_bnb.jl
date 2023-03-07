using SparseArrays
include("../mixed_binary_solver.jl")
#import toy_bnb.jl: solve_in_Clarabel, evaluate_constraint, check_lb_pruning, update_ub, select_leaf
include("toy_bnb.jl")

function add_branching_constraint_new(b::Vector, n::Int, integer_vars, fixed_x_indices, fix_values, upper_or_lower_vec)   
    
    println("Fixed x index: ",fixed_x_indices)
    if ~isnothing(fixed_x_indices) && ~isnothing(fix_values)
        # reminder: b is 1+2*n long where n is the TOTAL number of variables 
        for (i,j,k) in zip(fixed_x_indices, fix_values, upper_or_lower_vec)
            if k == 1
                println("set upper bound for index: ", i," to ", j)
                # this is for x[i] <= value which are in the last m:end elements of augmented b
                b[end-n+i] = j

            elseif k == -1
                println("set lower bound for index: ", i," to ", -j)
                # this is for x[i] >= value which are in the last 2m:m elements of augmented b
                b[end-2*n+i] = j 
            end
        end
            
        println("Updated b: ",b)
    end
    return b
end
function reset_b_vector(b::Vector,n::Int)
    n_x = Int(n/2) # number of x variables is half the total nb of vars
    b[2:end-n_x] = zeros(3*n_x)
    b[end-n_x+1:end] = ones(n_x)
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fix_x_values is the vector of the
variables of fixed_x_indices that are currently fixed to a boolean"
function compute_lb_new(solver, n::Int, fixed_x_indices, fix_x_values,integer_vars, upper_or_lower_vec, best_ub, early_num::Int,total_iter::Int, early_term_enable::Bool, warm_start::Bool, λ, prev_x= Nothing, prev_z=Nothing, prev_s = Nothing, total_nodes = Nothing)
    b = solver.data.b # so we modify the data field vector b directly, not using any copies of it
    if ~isnothing(fixed_x_indices)
        #relax all integer variables before adding branching bounds specific to this node
        reset_b_vector(b,n) 
    end
    simple_domain_propagation!(b,-b[1])
    b = add_branching_constraint_new(b,n,integer_vars,fixed_x_indices,fix_x_values,upper_or_lower_vec)
    

    #= println("cones : ", solver.cones.cone_specs)
    println(" Solver.variables.x : ", solver.variables.x)
    println(" Solver.variables.z : ", solver.variables.z)
    println(" Solver.variables.s : ", solver.variables.s)  =#

    #solve using IPM with early_termination checked at the end if feasible solution best_ub is available
    solution = solve_in_Clarabel(solver, best_ub, early_term_enable, warm_start,λ, prev_x, prev_z, prev_s)
    total_iter += solver.info.iterations
    total_nodes += 1
    if isnothing(solution)
        early_num = early_num+ 1
        printstyled("Node early termination, increase counter by 1 \n", color = :red)
        return Inf, [Inf for _ in 1:n],[Inf for _ in 1:n],[Inf for _ in 1:n], early_num, total_iter, total_nodes
    end
    if solution.status== Clarabel.SOLVED
        println("Values of relaxed solution ", solution.x)
        return solution.obj_val, solution.x, solution.z, solution.s, early_num, total_iter, total_nodes
    else 
        println("Infeasible or early terminated relaxed problem")
        return Inf, [Inf for _ in 1:n],[Inf for _ in 1:n],[Inf for _ in 1:n], early_num, total_iter, total_nodes
    end
end

""" base_solution is the first solution to the relaxed problem"""
function branch_and_bound_solve(solver, base_solution, n, ϵ, integer_vars=collect(1:n),pruning_enable::Bool=true, early_term_enable::Bool = true, warm_start::Bool = false, λ=0.0)
    #initialise global best upper bound on objective value and corresponding feasible solution (integer)
    best_ub = Inf 
    early_num = 0
    best_feasible_solution = zeros(n)
    node_queue = Vector{BnbNode}()
    max_nb_nodes = 500
    total_iter = 0
    total_nodes = 0
    fea_iter = 0
    if base_solution.status == Clarabel.SOLVED
        lb = base_solution.obj_val
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        best_ub, best_feasible_solution = compute_ub(solver, n,integer_vars, base_solution.x)
        # this is our root node of the binarytree
        root = BnbNode(ClarabelNodeData(solver,base_solution.x,base_solution.z, base_solution.s,[],[],[],lb)) #base_solution is node.data.Model
        node = root
        push!(node_queue,node)
        iteration = 0
        total_nodes += 1
        x = zeros(n)
    end
    

    # 3) start branching
    while ~isempty(node_queue) 
        if length(node_queue) >= max_nb_nodes
            printstyled("MAXIMUM NUMBER OF NODES REACHED \n",color = :red)
            break
        end 
        println(" ")
        println("Node queue length : ", length(node_queue))
        # pick and remove node from node_queue
        node = select_leaf(node_queue, best_ub)
        #node = splice!(node_queue,argmin(n.data.lb for n in node_queue))

        println("Difference between best ub: ", best_ub, " and best lb ",node.data.lb, " is ",best_ub - node.data.lb ) 
        if best_ub - node.data.lb <= ϵ
            break
        end
        printstyled("Best ub: ", best_ub, " with feasible solution : ", best_feasible_solution,"\n",color= :green)
        println("current node at depth ", node.data.depth, " has data.solution as ", node.data.solution_x)

        #IMPORTANT: the x should NOT change after solving in compute_lb_new or compute_ub -> use broadcasting
        x .= node.data.solution_x
        if x != node.data.solver.solution.x
            printstyled("x is not equal to solver.solution.x\n",color= :red)
        end
        # heuristic guessing for fractional solution: which edge to split along i.e. which variable to fix next? 
        fixed_x_index = pick_index(x, integer_vars, node.data.fixed_x_ind,false) 
        println("GOT BRANCHING VARIABLE: ", fixed_x_index)
        ceil_value = 1.0
        floor_value = 0.0
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always fixes the next variable to closest lower integer
        fixed_x_left = vcat(node.data.fixed_x_values, floor_value) 
        upper_or_lower_vec_left = vcat(node.data.upper_or_lower_vec, 1)
        println("fixed_x_left: ", fixed_x_left)

        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_solver = node.data.solver  
        #NOTE: if early terminated node, compute_lb_new returns Inf,Inf then check_lb_pruning prunes this node
        l̃, relaxed_x_left, z_left,s_left, early_num, total_iter, total_nodes = compute_lb_new(left_solver, n,fixed_x_indices, fixed_x_left, integer_vars, upper_or_lower_vec_left, best_ub, early_num, total_iter, early_term_enable, warm_start, λ, x, node.data.solution_z, node.data.solution_s, total_nodes) 
        println("solved for l̃: ", l̃)
        #create new child node (left)
        left_node = leftchild!(node, ClarabelNodeData(left_solver, relaxed_x_left,z_left,s_left, fixed_x_indices, fixed_x_left, upper_or_lower_vec_left, l̃)) 
        # prune node if l̄ > current ub or if l̄ = Inf
        if pruning_enable 
            check_lb_pruning(left_node,best_ub)
        end
        # only perform upper bound calculation if not pruned:
        if ~left_node.data.is_pruned
            ũ, feasible_x_left = compute_ub(left_solver, n,integer_vars,relaxed_x_left)
            println("Left node, solved for ũ: ", ũ)
            best_ub, best_feasible_solution, fea_iter = update_ub(ũ, feasible_x_left, best_ub, best_feasible_solution, left_node.data.depth,total_iter, fea_iter)
            push!(node_queue,left_node)
        end
        #println("DEBUG... fixed indices on left branch are : ", fixed_x_indices, " to fixed_x_left ", fixed_x_left)
        println(" ")
        
        # solve the right child problem to get l-bar and u-bar
        right_solver = node.data.solver
        fixed_x_right = vcat(node.data.fixed_x_values, -ceil_value) # NOTE: set to negative sign due to -x[i] + s = -b[i] if we want lower bound on x[i]
        upper_or_lower_vec_right = vcat(node.data.upper_or_lower_vec, -1)
        println("fixed_x_right: ", ceil(x[fixed_x_index]))
        l̄, relaxed_x_right, z_right, s_right, early_num, total_iter, total_nodes = compute_lb_new(right_solver,n, fixed_x_indices, fixed_x_right, integer_vars,upper_or_lower_vec_right, best_ub, early_num, total_iter, early_term_enable, warm_start,λ, x, node.data.solution_z, node.data.solution_s, total_nodes)
        println("solved for l̄: ", l̄)
        #create new child node (right)
        right_node = rightchild!(node, ClarabelNodeData(right_solver, relaxed_x_right, z_right, s_right, fixed_x_indices,fixed_x_right, upper_or_lower_vec_right, l̄))
        if pruning_enable
            check_lb_pruning(right_node, best_ub)
        end
        # only perform upper bound calculation if not pruned:
        if ~right_node.data.is_pruned
            ū, feasible_x_right = compute_ub(right_solver, n,integer_vars,relaxed_x_right)
            println("Right node, solved for ū: ", ū)
            best_ub, best_feasible_solution, fea_iter = update_ub(ū, feasible_x_right, best_ub, best_feasible_solution, right_node.data.depth,total_iter, fea_iter)
            push!(node_queue,right_node)
        end
        #println("DEBUG... fixed indices on right branch are : ", fixed_x_indices, " to ", fixed_x_right)        

        ind = 1
        while ind ≤ lastindex(node_queue)
            if check_lb_pruning(node_queue[ind],best_ub)
                printstyled("Fathom node in queue with lb > U!\n", color = :red)
                deleteat!(node_queue,ind)
            end
            ind += 1
        end
        iteration += 1
        println("iteration : ", iteration)
    end
    
    return best_ub, best_feasible_solution, early_num,total_iter, fea_iter, total_nodes
end
