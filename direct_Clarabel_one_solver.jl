using SparseArrays
include("mixed_binary_solver.jl")
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
function getAugmentedData(A::SparseMatrixCSC,b::Vector,cones::Vector,integer_vars::Vector,n::Int)
    m = length(integer_vars)
    # adding m rows of 0*x[i] <= 1
    Ã = sparse(collect(1:m),[i for i in integer_vars] ,-1*ones(m),m,n) #corresponding to lower bound constraints -x+s= -b
    Ā = sparse_vcat(A,Ã)
    b̄ = vcat(b,zeros(m)) 
    s̄ = vcat(cones,Clarabel.NonnegativeConeT(m))

    #=for _ in 1:m
        push!(s̄,Clarabel.NonnegativeConeT(1)) # prevent dimension mismatch when evaluating A*x+s in compute_ub()
    end=#
    return Ā,b̄, s̄
end
function reset_solver!(solver)
    n = solver.data.n
    solver.variables = Clarabel.DefaultVariables{Float64}(n, solver.cones)
    solver.residuals = Clarabel.DefaultResiduals{Float64}(n, solver.data.m)
    solver.info = Clarabel.DefaultInfo{Float64}()
    solver.prev_vars = Clarabel.DefaultVariables{Float64}(n, solver.cones)
    solver.solution = Clarabel.DefaultSolution{Float64}(solver.data.m,n)
end 

function solve_in_Clarabel(solver)
    # CRUCIAL: reset the solver info (termination status) and the solver variables when you use the same solver to solve an updated problem
    #reset_solver!(solver) #TODO: maybe not needed?
#    println("b in solver: ", solver.data.b)
    result = Clarabel.solve!(solver)

    return result
end


function add_branching_constraint(A::SparseMatrixCSC,b::Vector, integer_vars, fixed_x_indices, fix_values, bounds)    
    if ~isnothing(fixed_x_indices) && ~isnothing(fix_values)
        m = length(integer_vars)
        # match the indices to the indices in augmented vector b (which only augmented for integer_vars)
        # e.g. integer_vars = [1,2,5], fixed_x_indices=[1,5] then we want the 1st and 3rd element
        indices_in_b = [findfirst(x->x==i,integer_vars) for i in fixed_x_indices]
        for (i,j,k) in zip(indices_in_b, fix_values, bounds)
            if k == "ub"
                println("set upper bound for index: ", i," to ", j)
                # this is for x[i] <= value which are in the last m:end elements of augmented b
                b[end-m+i] = j
                A[end-m+i,end-m+i] = 1

            elseif k == "lb"
                println("set lower bound for index: ", i," to ", j)
                # this is for x[i] >= value
                A[end-m+i,end-m+i] = -1
                b[end-m+i] = -j # needs negative j for the NonnegativeConeT constraint
            end
        end
        
    end
    return A, b

end
function reset_Ab(A, b::Vector,integer_vars::Vector)
    m = length(integer_vars)
    A[collect(end-m+1:end),[i for i in integer_vars]]==sparse(Matrix(-1*I,2,2))
    b[end-m+1:end] = zeros(m)
end

function evaluate_constraint(solver,x)  
    # TODO: check miOSQP code (this is the heuristics part)
    cone_specs = solver.cones.cone_specs
    residual = zeros(length(solver.data.b))
    residual .= solver.data.b
    mul!(residual, solver.data.A, x, -1, 1)
    j = 1
    while j <= length(solver.data.b)
        for i in eachindex(cone_specs)
            t = typeof(cone_specs[i])
            k = j:j+cone_specs[i].dim-1
            
            if t == Clarabel.ZeroConeT
                println("Evaluating ", residual[k], " == 0 ?")
                if ~all(isapprox.(residual[k],0,atol = 1e-3))
                    return false
                end
                
            
            elseif t == Clarabel.NonnegativeConeT
                println("Evaluating ", residual[k], ">= 0 ?")

                if minimum(residual[k])< 0
                    return false
                    
                end

            end
            j = j+ cone_specs[i].dim
            

        end 
    end
    return true
end
" returns the upper bound computed when given the model as well as the rounded variable solution.
For Mixed_binary_solver: variables of indices from integer_vars (defaulted to all) are rounded based on relaxed_vars from lower_bound_model.
fixed_x_values is the vector of corresponding variables fixed on this iteration, if isnothing, that is the root case
and all variables take on the rounded values."
function compute_ub(solver,n::Int, integer_vars,relaxed_vars)
    # set the relaxed_vars / lb result equal to the rounded binary values
    # if these are in the set of binary variables   
    if isinf(relaxed_vars[1])
        return Inf, [Inf for _ in 1:n]
    end
    x = deepcopy(relaxed_vars) # TOASK
    P = solver.data.P
    q = solver.data.q
    
    for i in integer_vars
        x[i] = round(relaxed_vars[i]) 
    end
    println("rounded variables: ", x)

    
    if evaluate_constraint(solver,x)
        obj_val = 0.5*x'*P*x + q'*x 
        println("Valid upper bound : ", obj_val," using feasible x: ", x)
        return obj_val, x
    else 
        println("Infeasible or unbounded problem for ub computation")
        return Inf, [Inf for _ in 1:n]
    end
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fix_x_values is the vector of the
variables of fixed_x_indices that are currently fixed to a boolean"
function compute_lb(solver, n, fixed_x_indices, fix_x_values,bounds,integer_vars)
    A = solver.data.A
    b = solver.data.b # so we modify the data field vector b directly, not using any copies of it
    
    if ~isnothing(fixed_x_indices)
        #relax all integer variables before adding branching bounds specific to this node
        reset_Ab(A, b,integer_vars) 
    end
    A, b = add_branching_constraint(A,b,integer_vars,fixed_x_indices,fix_x_values,bounds)
    reset_solver!(solver) 
    println(" A : ",A)
    println(" b ", b)
    println("cones : ", solver.cones.cone_specs)
    println(" Solver.variables.x : ", solver.variables.x)
    println(" Solver.variables.z : ", solver.variables.z)
    println(" Solver.variables.s : ", solver.variables.s)



    solution = solve_in_Clarabel(solver)
    debug_b = deepcopy(solver.data.b)
    println("solution.x looks like: ",solution.x)
    if solution.status== Clarabel.SOLVED
        println("Values of relaxed solution ", solution.x)
        return solution, solution.obj_val, solution.x, debug_b
    else 
        println("Infeasible or unbounded problem for lb computation")
        println(" BAD SOLUTION : ", solution.x, solution.status)
        println(" Solver.variables.x : ", solver.variables.x)
        println(" Solver.variables.z : ", solver.variables.z)
        println(" Solver.variables.s : ", solver.variables.s)


        return solution, Inf, [Inf for _ in 1:n], debug_b
    end
end


""" base_solution is the first solution to the relaxed problem"""
function branch_and_bound_solve(solver, base_solution, n, ϵ, integer_vars=collect(1:n))
    
    if base_solution.status == Clarabel.SOLVED
        lb = base_solution.obj_val
        println("Solution x of unbounded base model: ", base_solution.x)
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        ub, feasible_x =compute_ub(solver, n,integer_vars, base_solution.x)
        term_status = "UNDEFINED"
        # this is our root node of the binarytree
        root = BnbNode(ClarabelNodeData(solver,base_solution,feasible_x,[],[],[],lb,ub)) #base_solution is node.data.Model
        root.data.debug_b = deepcopy(solver.data.b)
        node = root
        println("root has ub: ", root.data.ub)
        iteration = 0
        x = zeros(length(feasible_x))

    else 
        term_status = "INFEASIBLE"
    end
    

    # 3) start branching
    while term_status == "UNDEFINED" 
        println("current node at depth ", node.data.depth, " has data.solution.x as ", node.data.solution.x)
        #the relaxed solution from compute_lb, != solution_x

        #IMPORTANT: the x should NOT change after solving in compute_lb or compute_ub -> use broadcasting
        x .= node.data.solver.solution.x 
        println(" IS IT same object as node.data.solver.solution.x ???? ", x) # yes I think
        # which edge to split along i.e. which variable to fix next?
        fixed_x_index = get_next_variable_to_fix_to_integer(x, integer_vars, node.data.fixed_x_ind) 
        if fixed_x_index == -1
            term_status = "OPTIMAL"
            break
        end
        println("GOT BRANCHING VARIABLE: ", fixed_x_index, " SET SMALLER THAN FLOOR (left): ", floor(x[fixed_x_index]), " OR GREATER THAN CEIL (right)", ceil(x[fixed_x_index]))
        ceil_value = ceil(x[fixed_x_index])
        floor_value = floor(x[fixed_x_index])
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always fixes the next variable to closest lower integer
        fixed_x_left = vcat(node.data.fixed_x_values, floor_value) 
        
        println("fixed_x_left: ", fixed_x_left)
        bounds_left = vcat(node.data.bounds, "ub")
        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_solver = node.data.solver  
        lb_solution,l̃, relaxed_x_left, debug_b = compute_lb(left_solver, n,fixed_x_indices, fixed_x_left, bounds_left,integer_vars) 
        println("solved for l̃: ", l̃)
        println("x after solving for l̃: ",x) # has not changed
        println(" data.solution.x as ", node.data.solution.x) #has changed! IDK why it has these particular values for x when solution status is insuff.progress

        ũ, feasible_x_left= compute_ub(left_solver, n,integer_vars,relaxed_x_left)
        println("solved for ũ: ", ũ)

        println("fixed indices on left branch are : ", fixed_x_indices, " to ", fixed_x_left)
        
        #create new child node (left)
        
        left_node = leftchild!(node, ClarabelNodeData(left_solver, lb_solution, feasible_x_left, fixed_x_indices, fixed_x_left,bounds_left,l̃,ũ)) 
        left_node.data.debug_b = debug_b

        # solve the right child problem to get l-bar and u-bar
        right_solver = node.data.solver
        fixed_x_right = vcat(node.data.fixed_x_values, ceil_value) 
        println("fixed_x_right: ", ceil(x[fixed_x_index]))
        bounds_right = vcat(node.data.bounds, "lb")
        #TODO: PROBLEM HERE
        lb_solution_right, l̄, relaxed_x_right,debug_b = compute_lb(right_solver,n, fixed_x_indices, fixed_x_right, bounds_right,integer_vars)
        println("solved for l̄: ", l̄)
        println("x after solving for l̄: ",x) #TODO: PROBLEM HERE


        ū, feasible_x_right = compute_ub(right_solver, n, integer_vars, relaxed_x_right)
        println("solved for ū: ", ū)

        println("fixed indices on right branch are : ", fixed_x_indices, " to ", fixed_x_right)
        #create new child node (right)
        right_node = rightchild!(node, ClarabelNodeData(right_solver, lb_solution_right, feasible_x_right, fixed_x_indices,fixed_x_right,bounds_right,l̄,ū))
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
        compute_lb(node.data.solver,n, node.data.fixed_x_ind, node.data.fixed_x_values, node.data.bounds,integer_vars) 

        println("Difference: ", root.data.ub, " - ",root.data.lb, " is ",root.data.ub-root.data.lb )
        println(" ")
        term_status = termination_status_bnb(root.data.ub,root.data.lb, ϵ)
        iteration += 1
        println("iteration : ", iteration)
    end
    #return iteration,Aleft,bleft,conesleft,Aleft2,bleft2,conesleft2
    
    return root, term_status
end


function main_Clarabel()
    n = 2
    k = 3
    m= 2
    integer_vars = sample(1:n, m, replace = false)
    sort!(integer_vars)
    Q = Matrix{Float64}(I, n, n) 
    Random.seed!(1234)
    c = rand(Float64,n)
    ϵ = 0.00000001
        # check against Gurobi
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

    optimizer = Clarabel.Optimizer
    old_model = build_unbounded_base_model(optimizer,n,k,Q,c)
    solve_base_model(old_model,integer_vars)

    #solve in Clarabel the relaxed problem

    
    P,q,A,b, cones= getClarabelData(old_model)
    Ā,b̄, s̄= getAugmentedData(A,b,cones,integer_vars,n)
    settings = Clarabel.Settings(verbose = true, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ā, b̄, s̄, settings)
    println(" A : ",solver.data.A)
    println(" b ", solver.data.b)
    
    result = Clarabel.solve!(solver)


    #start bnb loop
    println("STARTING CLARABEL BNB LOOP ")
    root, term_status = branch_and_bound_solve(solver, result,n,ϵ, integer_vars)
    println("Termination status of Clarabel bnb:" , term_status)
    println("Found objective: ", root.data.ub, " using ", round.(root.data.solution_x,digits=3))
    println("Compare with exact: ", round(norm(root.data.solution_x - value.(exact_model[:x]))),round(root.data.ub-objective_value(exact_model)))
    println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 

    
    return solver
end