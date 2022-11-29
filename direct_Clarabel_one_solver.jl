using SparseArrays
include("mixed_integer_solver.jl")
function getClarabelData(model::Model)
    # access the Clarabel solver object
    solver = JuMP.unsafe_backend(model).solver
    # now you can get data etc
    data = solver.data
    P = data.P
    q = data.q
    A = data.A
    b = data.b
    s = solver.cones.cone_specs
    return P,q,A,b,s
end
function getAugmentedData(P,q,A,b,cones,integer_vars)
    m = length(integer_vars)
    A2 = sparse(collect(1:m),[i for i in integer_vars] ,-1* ones(m))
    A3 = sparse(collect(1:m),[i for i in integer_vars] ,-1* ones(m))
    A = sparse_vcat(A,A2, A3)
    b = vcat(b,zeros(n))
    for i = 1:n
        push!(cones,Clarabel.NonnegativeConeT(1)) # this is so that we can modify it to ZeroconeT afterwards
    end
    return P,q,A,b, cones
end
function reset_solver!(solver)
    solver.variables = Clarabel.DefaultVariables{Float64}(n, solver.cones)
    solver.residuals = Clarabel.DefaultResiduals{Float64}(n, solver.data.m)
    solver.info = Clarabel.DefaultInfo{Float64}()
    solver.prev_vars = Clarabel.DefaultVariables{Float64}(n, solver.cones)
    solver.solution = Clarabel.DefaultSolution{Float64}(solver.data.m,n)
end
function solve_in_Clarabel(solver)
    # CRUCIAL: reset the solver info (termination status) and the solver variables when you use the same solver to solve an updated problem
    reset_solver!(solver) #TODO: maybe not needed?
    # solver.data.b .= b # !!!!! ".=" overwrites each element of b (broadcast "=" operator) instead of pointing to another object!!!
    println("Solver variables inside solve_in_Clarabel(): ", solver.variables)
    println("A in solver: ", solver.data.A)
    println("b in solver: ", solver.data.b)
    println("cones in solver: ", solver.cones)

    result = Clarabel.solve!(solver)

    return result
end


function add_branching_constraint(A, b::Vector, n::Int, fixed_x_indices, fix_values, bounds)    
    if ~isnothing(fixed_x_indices) && ~isnothing(fix_values)
        println("add_branching_constraint for indices: ", fixed_x_indices)
        for (i,j,k) in zip(fixed_x_indices, fix_values, bounds)
            if k == "ub"
                println("Setting ub of variable ",i," to ", j)
                # this is for x[i] <= value
                A[end-n+i,end-n+i] = 1
                b[end-n+i] = j
            elseif k == "lb"
                println("Setting lb of variable ",i," to ", j)
                # this is for x[i] >= value since we have Clarabel.NonnegativeConeT(1)
                A[end-n+i,end-n+i] = -1
                b[end-n+i] = -j
            end
        end
            
        
    end
    return A, b

end
function relax_vars(A,b,cones,n,fixed_x_indices)
    for i in 1:n
        if ~(i in fixed_x_indices) 
            b[end-n+i] = 0
            A[end-n+i, end-n+i] = -1
            cones[end-n+i]= Clarabel.NonnegativeConeT(1)
        end
    end
end
### OVERWRITE THE FOLLOWING FUNCTIONS FROM BINARY_SOLVER.JL
" returns the upper bound computed when given the model as well as the rounded variable solution.
For Mixed_binary_solver: variables of indices from integer_vars (defaulted to all) are rounded based on relaxed_vars from lower_bound_model.
fixed_x_values is the vector of corresponding variables fixed on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub(solver,n::Int, integer_vars,fixed_x_indices, fix_x_values, bounds,relaxed_vars)
    solver = deepcopy(solver)
    A = solver.data.A
    b = solver.data.b # so we modify the data field vector b directly, not using any copies of it
    cones = solver.cones.cone_specs

    # crucial step: relax any variables not in these vectors is currently fixed (i.e. b[i+1]==b[i+1+n])
    # (if we changed to a different branch but solver is same for all nodes)
    if ~isnothing(fixed_x_indices)
        relax_vars(A,b,cones,n,fixed_x_indices) 
    end

    # set the relaxed variables equal to the rounded binary values
    # if these are in the set of binary variables   
    rounded_bounds =  [round(value(relaxed_vars[i])) for i in integer_vars]
    println("rounded bounds vector: ", rounded_bounds)

    for (index,j) = zip(integer_vars, 1:lastindex(rounded_bounds))
        println("ROUNDING VARIABLE ", index, " TO ", rounded_bounds[j])
        A[end-n+index,end-n+index] = -1
        b[end-n+index] = -rounded_bounds[j] #TODO what about this?
        cones[end-n+index]= Clarabel.ZeroConeT(1)# this is for x[index] == value
    end
    #println("B before add_branching_constraint after rounding:", b,cones)
    # force the branching variables to fixed value
    A, b, cones = add_branching_constraint(A,b,cones,n,fixed_x_indices,fix_x_values, bounds)
    #println("After adding fixed vars b and s: ", b,cones)
    debug_b = deepcopy(solver.data.b)
    solution = solve_in_Clarabel(solver) 
    if solution.status== Clarabel.SOLVED
        println("Values of upper bound solution (feasible)", solution.x)
        return solution.obj_val, solution.x, debug_b, A,b, cones
    else 
        println("Infeasible or unbounded problem for ub computation")
        return Inf, [Inf for _ in 1:n], debug_b,A,b,cones
    end
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fix_x_values is the vector of the
variables of fixed_x_indices that are currently fixed to a boolean"
function compute_lb(solver, n, fixed_x_indices, fix_x_values)
    A = solver.data.A
    b = solver.data.b # so we modify the data field vector b directly, not using any copies of it
    cones = solver.cones.cone_specs
    if ~isnothing(fixed_x_indices)
        relax_vars(A,b,cones,n,fixed_x_indices) 
    end
    A, b, cones = add_branching_constraint(A,b,cones,n,fixed_x_indices,fix_x_values)
    
    println("Solver vars before solve in compute lb: ", solver.variables)
    solution = solve_in_Clarabel(solver)
    
    if solution.status== Clarabel.SOLVED
        println("Values of relaxed solution ", solution.x)
        return solution, solution.obj_val, solution.x, A,b,cones
    else 
        println("Infeasible or unbounded problem for lb computation")
        return solution, Inf, [Inf for _ in 1:n],A,b,cones
    end
end


""" base_solution is the first solution to the relaxed problem"""
function branch_and_bound_solve(solver, base_solution, n, ϵ, integer_vars=collect(1:n))
    
    if base_solution.status == Clarabel.SOLVED
        lb = base_solution.obj_val
        println("Solution x of unbounded base model: ", base_solution.x)
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        ub, feasible_x, debug_b =compute_ub(solver, n,integer_vars, nothing, nothing, base_solution.x)
        term_status = "UNDEFINED"
    else 
        term_status = "INFEASIBLE"
    end
    # this is our root node of the binarytree
    root = BinaryNode(ClarabelNodeData(solver,base_solution,feasible_x,[],[],lb,ub)) #base_solution is node.data.Model
    root.data.debug_b = debug_b
    node = root
    println("root has ub: ", root.data.ub)
    iteration = 0

    # 3) start branching
    while term_status == "UNDEFINED" 
        println("current node at depth ", node.data.depth, " has x as ", node.data.solution.x)
        x = node.data.solution.x #the relaxed solution from compute_lb, != solution_x
        # which edge to split along i.e. which variable to fix next?
        fixed_x_index = get_next_variable_to_fix_to_integer(x, integer_vars, node.data.fixed_x_ind) 
        println("GOT NEXT VAR TO FIX: ", fixed_x_index, " TO FLOOR : ", floor(x[fixed_x_index]), " TO CEIL ", ceil(x[fixed_x_index]))
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always fixes the next variable to closest lower integer
        fixed_x_left = vcat(node.data.fixed_x_values, floor(x[fixed_x_index])) 
        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_solver = node.data.solver  
        println("")
        lb_solution,l̃, relaxed_x_left,Aleft,bleft,conesleft = compute_lb(left_solver, n,fixed_x_indices, fixed_x_left) 
        println("solved for l̃: ", l̃)
        ũ, feasible_x_left, debug_b, Aleft2, bleft2,conesleft2= compute_ub(left_solver, n,integer_vars,fixed_x_indices, fixed_x_left, relaxed_x_left)
        println("solved for ũ: ", ũ)
        println("fixed indices on left branch are : ", fixed_x_indices, " to ", fixed_x_left)
        
        #create new child node (left)
        
        left_node = leftchild!(node, ClarabelNodeData(left_solver, lb_solution, feasible_x_left, fixed_x_indices, fixed_x_left,l̃,ũ)) 
        left_node.data.debug_b = debug_b

        # solve the right child problem to get l-bar and u-bar
        right_solver = node.data.solver
        fixed_x_right = vcat(node.data.fixed_x_values, ceil(x[fixed_x_index])) 
        lb_solution_right, l̄, relaxed_x_right,Ar,br,conesr = compute_lb(right_solver,n, fixed_x_indices, fixed_x_right)
        
        println("solved for l̄: ", l̄)
        ū, feasible_x_right, debug_b,Ar2,br2,conesr2= compute_ub(right_solver, n, integer_vars, fixed_x_indices, fixed_x_right, relaxed_x_right)
        println("solved for ū: ", ū)
        println("fixed indices on right branch are : ", fixed_x_indices, " to ", fixed_x_right)
        #create new child node (right)
        right_node = rightchild!(node, ClarabelNodeData(right_solver, lb_solution_right, feasible_x_right, fixed_x_indices,fixed_x_right,l̄,ū))
        right_node.data.debug_b = debug_b

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

        # CRUCIAL: solve again to get the correct relaxed solution (see line 2 after "while") 
        # but note that feasible x solution stored in data.solution_x
        compute_lb(node.data.solver,n, node.data.fixed_x_ind, node.data.fixed_x_values) 

        println("Difference: ", root.data.ub, " - ",root.data.lb, " is ",root.data.ub-root.data.lb )
        println(" ")
        term_status = termination_status_bnb(root.data.ub,root.data.lb, ϵ)
        iteration += 1
        println("iteration : ", iteration)
    end
    #return iteration,Aleft,bleft,conesleft,Aleft2,bleft2,conesleft2
    
    return root, term_status
end




old_model = build_unbounded_base_model(optimizer,n,k,Q,c)
solve_base_model(old_model,integer_vars)

#solve in Clarabel the relaxed problem

# NOTE: b and solver.status must be reset at each bnb iteration, P,q,A remain 
# PROBLEM 25/11: lb AND ub are wrong when setting variable to 0.0, constraints are ignored
# both for setting x[2] to 0.0 and for ROUNDING x's to the rounded values!!
# check info status to find the point where it diverges with JuMP model
P,q,A,b, cones= getClarabelData(old_model)
P,q,A,b, cones= getAugmentedData(P,q,A,b,cones,n)
settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
solver   = Clarabel.Solver()

Clarabel.setup!(solver, P, q, A, b, cones, settings)

result = Clarabel.solve!(solver)
println("INITIAL VARS: ", solver.variables)


# start bnb loop
println("STARTING CLARABEL BNB LOOP ")
#i,Aleft,bleft,conesleft,Aleft2,bleft2,conesleft2 = branch_and_bound_solve(solver, result,n,ϵ, integer_vars)
root, term_status = branch_and_bound_solve(solver, result,n,ϵ, integer_vars)
println("Found objective: ", root.data.ub, " using ", root.data.solution_x)