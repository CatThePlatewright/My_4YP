using SparseArrays
using TimerOutputs
include("mixed_binary_solver.jl")

"""domain propagation for MIMPC in IPM format:
min ...
s.t. Ãx ≤ b̃ 
where Ã = vcat(A, -A[m+1:end,:], -I, I), m = length(u) # ATTENTION: to be consistent with toy problem, have ordered 1. lb 2.ub
b̃ = vcat(u, -l[m+1:end], -lb, ub)        # ATTENTION: true vector l is only 6=12:18(length of extracted l) long, note: u is 18 long for N=2, u+l has length 12*N
where A is of size (m/2)xn and I is nxn

"""
function domain_propagation_mpc!(Ã,b,n,i_idx)
    m = Int(3*n / 2) # find how long the original vector u is since we concatenated u into b in getClarabelData()
    A= Ã[1:m,:]
    u = b[1:m]
    l = vcat(-infinity*ones(n), -b[m+1:end-2*n] ) #concatenate n times -inf's (paddings) with true lower bounds l (only 6-element long for N=2) -> gives length of m
    lb = -b[end-2*n+1:end-n] 
    ub = b[end-n+1:end]

    flag = true #check whether domain propagation is effective

    while flag
        pre_lb = deepcopy(lb)
        pre_ub = deepcopy(ub)

        for row = 1:m
            #find nonzero items in each constraint
            column, value = findnz(A[row,:])

            for i = 1:lastindex(column)
                upper = u[row] # this is u_b in paper
                lower = l[row]
                for j = 1:lastindex(column)
                    #except the index i
                    if i == j
                        continue
                    end
                    #check the sign of coefficient, e.g. -1*xi <= -l for xi>= l
                    if value[j] > 0 
                        upper -= value[j]*lb[column[j]]
                        if lower != -infinity  # no need to update lower bounds when it is -Inf since it is a padding only
                            lower -= value[j]*ub[column[j]]
                        end
                    else
                        upper -= value[j]*ub[column[j]]
                        if lower != infinity
                            lower -= value[j]*lb[column[j]]
                        end
                    end
                end

                #update bound
                if value[i] > 0
                    ub[column[i]] = min(ub[column[i]], upper/value[i])
                    lb[column[i]] = max(lb[column[i]], lower/value[i])
                else
                    ub[column[i]] = min(ub[column[i]], lower/value[i])
                    lb[column[i]] = max(lb[column[i]], upper/value[i])
                end
                #tighten bound for integer variables
                @. ub[i_idx] = floor(ub[i_idx])
                @. lb[i_idx] = ceil(lb[i_idx])
            end
            

        end
        if ub != pre_ub 
            println("domain_propagation_mpc: updated upper bounds", ub)
        end
        if lb != pre_lb
            println("domain_propagation_mpc: updated lower bounds", lb)
        end

        #domain propogation ends (convergence)
        if (lb == pre_lb) && (ub == pre_ub)
            println("domain_propagation_mpc: convergence")
            flag = false
        end
        
    end

    #detect infeasibility, return immediately
    if (lb .<= ub)!=ones(length(lb))
        println("Detect primal infeasibility in domain propagation")
        println(ub)
        println(lb)
        return Inf
    end
    b .= vcat(u, -l[n+1:end], -lb, ub)
    return 1
end

function add_branching_constraint(b::Vector, integer_vars, fixed_x_indices, fix_values, upper_or_lower_vec, debug_print=false)    
    if ~isnothing(fixed_x_indices) && ~isnothing(fix_values)
        m = length(integer_vars)
        # match the indices to the indices in vector b that only contains box constraints for u vars not the x vars! (checked on 05/03)
        # e.g. integer_vars = [1,2,5], fixed_x_indices=[1,5] then we want the 1st and 3rd element
        indices_in_b = [findfirst(x->x==i,integer_vars) for i in fixed_x_indices]
        for (i,j,k) in zip(indices_in_b, fix_values, upper_or_lower_vec)
            if k == 1
                if debug_print
                    println("set upper bound for index: ", i," to ", j)
                end
                # this is for x[i] <= value which are in the last m:end elements of augmented b
                b[end-m+i] = j
            elseif k == -1
                if debug_print
                    println("set lower bound for index: ", i," to ", -j)
                end
                # this is for x[i] >= value which are in the last 2m:m elements of augmented b
                b[end-2*m+i] = j # is already passed as negative l in x[i] >= l
            end
        end
        # println(b)
        
    end
    return b
end
function reset_b_vector(b::Vector,integer_vars::Vector)
    m = length(integer_vars)
    b[end-2*m+1:end-m]=ones(m) # previous last 2m rows correspond to lb on x in the form of -x<= -lb
    b[end-m+1:end] = ones(m) #last 2m rows correspond to ub on x
end

" return the lower bound as well as the values of x computed (for use by compute_ub()).
model is given with relaxed constraints. fix_x_values is the vector of the
variables of fixed_x_indices that are currently fixed to a boolean"
function compute_lb(solver, n::Int, fixed_x_indices, fix_x_values,integer_vars, upper_or_lower_vec, best_ub, early_num::Int, 
    total_iter::Int, total_time::Float64, early_term_enable::Int, warm_start::Bool, λ, luS,prev_x= Nothing, prev_z=Nothing, prev_s = Nothing,debug_print::Bool=false,dom_prog_enable::Bool=false, total_nodes=Nothing)
    A = solver.data.A
    b = solver.data.b # so we modify the data field vector b directly, not using any copies of it
    if ~isnothing(fixed_x_indices)
        #relax all integer variables before adding branching bounds specific to this node
        reset_b_vector(b,integer_vars) 
    end
    
    b = add_branching_constraint(b,integer_vars,fixed_x_indices,fix_x_values,upper_or_lower_vec, debug_print)
    if dom_prog_enable
        domain_prop = domain_propagation_mpc!(A,b,n,integer_vars)
        if isinf(domain_prop)
            println("Infeasible relaxed problem")
            return Inf, [Inf for _ in 1:n], [Inf for _ in 1:n],[Inf for _ in 1:n],early_num, total_iter, total_time, fact_time, solve_time, total_nodes
        end 
    end

    #solve using IPM with early_termination checked at the end if feasible solution best_ub is available
    solution = solve_in_Clarabel(solver, best_ub, early_term_enable, warm_start, λ, prev_x, prev_z, prev_s,luS,debug_print)
    total_iter += solver.info.iterations
    # total_time += solver.info.solve_time
    fact_time = TimerOutputs.time(solver.timers["solve!"]["IP iteration"]["kkt update"])/1e9
    solve_time = TimerOutputs.time(solver.timers["solve!"]["IP iteration"]["kkt solve"])/1e9
    total_time += fact_time + solve_time
    total_nodes += 1

    # printstyled("An IPM takes time ", total_time, " s \n")
    # printstyled("Fact time ", fact_time, " ms   and ", "Solve time", solve_time, " ms  \n")
    if isnothing(solution)
        early_num = early_num+ 1
        # early_time = TimerOutputs.time(solver.timers["solve!"]["IP iteration"]["early termination"])/1e9
        # printstyled("Node early termination takes ", early_time/solver.info.solve_time*100, " \\% percentage \n", color = :red)
        # printstyled("Node early termination \n", color = :red)
        return Inf, [Inf for _ in 1:n],[Inf for _ in 1:n],[Inf for _ in 1:n], early_num, total_iter, total_time, fact_time, solve_time, total_nodes
    end
    if solution.status== Clarabel.SOLVED
        if debug_print
            println("Values of relaxed solution ", solution.x)
        end
        return solution.obj_val, solution.x,solution.z, solution.s, early_num, total_iter, total_time, fact_time, solve_time, total_nodes
    else 
        println("Infeasible relaxed problem")
        return Inf, [Inf for _ in 1:n], [Inf for _ in 1:n],[Inf for _ in 1:n],early_num, total_iter, total_time, fact_time, solve_time, total_nodes
    end
end


function solve_in_Clarabel(solver, best_ub, early_term_enable::Int, warm_start::Bool,λ, prev_x, prev_z, prev_s,luS, debug_print::Bool)
    result = Clarabel.solve!(solver, best_ub, early_term_enable, warm_start, luS, debug_print,λ, prev_x, prev_z, prev_s)
    # println("node cost is ", solver.solution.obj_val)
    return result
end

function evaluate_constraint_mpc(solver,x, integer_vars, luS)
    if luS !== nothing
        g_width = solver.cones.cone_specs[1].dim # first elements of Ax+s=b encode Gx==h
        G = solver.data.A[1:g_width,:] 
        numel_states = length(x)-length(integer_vars)
        Gx = G[:,1:end-length(integer_vars)] # entries corresponding to continuous state vars
        if size(Gx)[1] != size(Gx)[2]
            error("Gx is not square!")
        end
        Gu = G[:,end - length(integer_vars) + 1:end] # entries corresponding to discrete input vars
        # fix up the continuous vars (x) to satisfy equality constraint (state dynamics)
        x[1:end-length(integer_vars)] .= inv(Matrix(Gx))*(solver.data.b[1:g_width] - Gu*x[end - length(integer_vars) + 1:end])
    end
    s = zeros(length(solver.data.b))
     #Same as:  residuals.rz_inf .=  data.b - data.A * variables.x 
    s .= solver.data.b
    mul!(s, solver.data.A, x, -1, 1) 
    # printstyled("Tau is : ", solver.variables.τ,"\n", color = :light_green)

    cone_specs = solver.cones.cone_specs
    # println(cone_specs, length(cone_specs))
    j = 1
    while j <= length(solver.data.b)
        for i in eachindex(cone_specs)
            t = typeof(cone_specs[i])
            k = j:j+cone_specs[i].dim-1
            if t == Clarabel.ZeroConeT 
                # should all be 0 by above construction
                if ~all(isapprox.(s[k],0,atol = 1e-7))
                    # println("ZeroConeT constraint not satisfied for s[k]: ",s[k])
                    return false
                end
            else
                z̃ = Clarabel.unit_margin(solver.cones[i],s[k],Clarabel.PrimalCone)
                if round(z̃,digits=7) < 0 
                    # println("NonnegativeConeT or SOC constraint not satisfied for s[k]: ",s[k])
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
function compute_ub(solver,n::Int,integer_vars,relaxed_vars,luS,debug_print=false)
    # set the relaxed_vars / lb result equal to the rounded binary values
    # if these are in the set of binary variables   
    if isinf(relaxed_vars[1])
        return Inf, [Inf for _ in 1:n]
    end
    x = deepcopy(relaxed_vars) # TOASK
    P = solver.data.P #this is only the upper triangular part
    q = solver.data.q
    for i in integer_vars
        x[i] = round(relaxed_vars[i]) 
    end
    if debug_print
        println("rounded variables: ", x[end - length(integer_vars) + 1:end])
    end

    if evaluate_constraint_mpc(solver,x,integer_vars,luS)
        obj_val = 0.5*x'*Symmetric(P)*x + q'*x 
        # println("Valid upper bound : ", obj_val," using integer feasible u: ", x[end - length(integer_vars) + 1:end])
        return obj_val, x
    else 
        # println("Infeasible or unbounded problem for ub computation")
        return Inf, [Inf for _ in 1:n]
    end
end

function check_lb_pruning(node, best_ub)
    #println("DEBUG node.data.lb - best_ub: ", node.data.lb, " - ", best_ub, " = ", node.data.lb - best_ub)
    if node.data.lb - best_ub >1e-5 || node.data.lb == Inf
        # println("Prune node with lower bound larger than best ub or ==INF")
        node.data.is_pruned = true
        return true
    end
    return false
end

function update_ub(u, feasible_solution, best_ub, best_feasible_solution, depth, 
    total_iter::Int, fea_iter::Int, 
    total_time::Float64, fea_time::Float64,
    total_fact_time::Float64, fea_fact_time::Float64,
    total_solve_time::Float64, fea_solve_time::Float64,
    total_nodes::Int,fea_nodes::Int)
    if (u < best_ub) # this only happens if node is not pruned
        if isinf(best_ub)
            fea_iter = total_iter
            fea_nodes = total_nodes
            fea_time = total_time
            fea_fact_time = total_fact_time
            fea_solve_time = total_solve_time
        end
        best_ub = u
        # println("FOUND BETTER UB AT DEPTH ", depth)
        best_feasible_solution = feasible_solution
    end
    return best_ub, best_feasible_solution, fea_iter, fea_time, fea_fact_time, fea_solve_time, fea_nodes
end
#select a leaf from leaves for computing
function select_leaf(node_queue::Vector{BnbNode}, best_ub)
    #depth first until find the first feasible solution
    if best_ub == Inf
        depth_set = []
        for node in node_queue
            push!(depth_set, node.data.depth)
        end
        depth = maximum(depth_set)
        max_set = findall(x -> x == depth, depth_set)

        index = 1
        #Find the one with the lowest bound
        lower_bound = Inf
        for i = 1:lastindex(max_set)
            if node_queue[max_set[i]].data.lb < lower_bound
                index = i
            end
        end 

        return splice!(node_queue, max_set[index])    #delete and return the selected leaf
    #best bound when we have a feasible solution
    else
        return splice!(node_queue,argmin(n.data.lb for n in node_queue))    #delete and return the selected leaf
    end
end
""" base_solution is the first solution to the relaxed problem"""
function branch_and_bound_solve(horizon_i, solver, base_solution, n, ϵ, integer_vars=collect(1:n),pruning_enable::Bool=true, early_term_enable::Int=0, warm_start::Bool = false, λ=0.0,luS = Nothing, debug_print::Bool = false, dom_prog_enable::Bool=false)
    #initialise global best upper bound on objective value and corresponding feasible solution (integer)
    best_ub = Inf 
    early_num = 0
    best_feasible_solution = zeros(n)
    node_queue = Vector{BnbNode}()
    max_nb_nodes = 1000
    total_iter = 0
    fea_iter = 0
    total_time = zero(Float64)
    fea_time = zero(Float64)
    total_fact_time = zero(Float64)
    fea_fact_time = zero(Float64)
    total_solve_time = zero(Float64)
    fea_solve_time = zero(Float64)
    if base_solution.status == Clarabel.SOLVED
        lb = base_solution.obj_val
        # println("Integer solution u of unbounded base model: ", base_solution.x[end - length(integer_vars) + 1:end])
        # 2) compute U1, upper bound on p* by rounding the solution variables of 1)
        best_ub, best_feasible_solution = compute_ub(solver, n,integer_vars, base_solution.x,luS)


        # this is our root node of the binarytree
        root = BnbNode(ClarabelNodeData(solver,base_solution.x,base_solution.z, base_solution.s,[],[],[],lb)) #base_solution is node.data.Model
        node = root
        push!(node_queue,node)
        iteration = 0
        fea_nodes = 1
        total_nodes = 1
        x = zeros(n)
    end
    

    # 3) start branching
    while ~isempty(node_queue) 
        if length(node_queue) >= max_nb_nodes
            printstyled("MAXIMUM NUMBER OF NODES REACHED \n",color = :red)
            break
        end 
        # println(" ")
        # println("Node queue length : ", length(node_queue))
        
        # pick and remove node from node_queue
        node = select_leaf(node_queue, best_ub)


        # println("Difference between best ub: ", best_ub, " and best lb ",node.data.lb, " is ",best_ub - node.data.lb ) 
        if best_ub - node.data.lb < ϵ
            break
        end
        # printstyled("Best ub: ", best_ub, " with feasible solution : ", best_feasible_solution,"\n",color= :green)
        if debug_print
            println("current node at depth ", node.data.depth, " has data.solution as ", node.data.solution_x[end - length(integer_vars) + 1:end])
        end
        
        #IMPORTANT: the x should NOT change after solving in compute_lb or compute_ub -> use broadcasting
        x .= node.data.solution_x
        if x != node.data.solver.solution.x && debug_print
            printstyled("x is not equal to solver.solution.x\n",color= :red)
        end
        # heuristic guessing for fractional solution: which edge to split along i.e. which variable to fix next? 
        fixed_x_index = pick_index(x, integer_vars, node.data.fixed_x_ind, debug_print) 
        if fixed_x_index == -1
            println("no remaining_branching_vars left at horizon ", horizon_i)
            error("stop")
        end
        if debug_print
            println("GOT BRANCHING VARIABLE: ", fixed_x_index, " SET SMALLER THAN FLOOR (left): ", floor(x[fixed_x_index]), " OR GREATER THAN CEIL (right)", ceil(x[fixed_x_index]))
        end
        ceil_value = ceil(x[fixed_x_index])
        floor_value = floor(x[fixed_x_index])
        fixed_x_indices = vcat(node.data.fixed_x_ind, fixed_x_index)
        # left branch always fixes the next variable to closest lower integer
        fixed_x_left = vcat(node.data.fixed_x_values, floor_value) 
        upper_or_lower_vec_left = vcat(node.data.upper_or_lower_vec, 1)
        # println("fixed_x_left: ", fixed_x_left)

        # solve the left child problem with 1 more fixed variable, getting l-tilde and u-tilde
        left_solver = node.data.solver  
        #NOTE: if early terminated node, compute_lb returns Inf,Inf then check_lb_pruning prunes this node
        l̃, relaxed_x_left, z_left,s_left, early_num, total_iter, total_time, fact_time, solve_time, total_nodes = compute_lb(left_solver, n,fixed_x_indices, fixed_x_left, integer_vars, upper_or_lower_vec_left, best_ub, early_num, total_iter, total_time, early_term_enable,warm_start, λ, luS, x, node.data.solution_z, node.data.solution_s, debug_print,dom_prog_enable, total_nodes) 
        total_fact_time += fact_time
        total_solve_time += solve_time
        
        # println("solved for l̃: ", l̃)
        #create new child node (left)
        left_node = leftchild!(node, ClarabelNodeData(left_solver, relaxed_x_left,z_left,s_left, fixed_x_indices, fixed_x_left, upper_or_lower_vec_left, l̃)) 
        # prune node if l̄ > current ub or if l̄ = Inf
        if pruning_enable 
            check_lb_pruning(left_node,best_ub)
        end
        # only perform upper bound calculation if not pruned:
        if ~left_node.data.is_pruned
            ũ, feasible_x_left = compute_ub(left_solver, n,integer_vars,relaxed_x_left, luS,debug_print)
            # println("Left node, solved for ũ: ", ũ)
            best_ub, best_feasible_solution, fea_iter, fea_time, fea_fact_time, fea_solve_time, fea_nodes = update_ub(ũ, feasible_x_left, best_ub, best_feasible_solution, left_node.data.depth, total_iter, fea_iter, total_time, fea_time, total_fact_time, fea_fact_time, total_solve_time, fea_solve_time, total_nodes,fea_nodes)
            push!(node_queue,left_node)
        end
        if debug_print
            println("fixed indices on left branch are : ", fixed_x_indices, " to fixed_x_left ", fixed_x_left)
        end
        # println(" ")
        
        # solve the right child problem to get l-bar and u-bar
        right_solver = node.data.solver
        fixed_x_right = vcat(node.data.fixed_x_values, -ceil_value) # NOTE: set to negative sign due to -x[i] + s = -b[i] if we want lower bound on x[i]
        upper_or_lower_vec_right = vcat(node.data.upper_or_lower_vec, -1)
        # println("fixed_x_right: ", ceil(x[fixed_x_index]))
        l̄, relaxed_x_right, z_right, s_right, early_num, total_iter, total_time, fact_time, solve_time, total_nodes= compute_lb(right_solver,n, fixed_x_indices, fixed_x_right, integer_vars,upper_or_lower_vec_right, best_ub, early_num, total_iter, total_time, early_term_enable,warm_start,λ,luS, x, node.data.solution_z, node.data.solution_s, debug_print,dom_prog_enable, total_nodes)
        total_fact_time += fact_time
        total_solve_time += solve_time
        
        # println("solved for l̄: ", l̄)
        #create new child node (right)
        right_node = rightchild!(node, ClarabelNodeData(right_solver, relaxed_x_right,z_right,s_right, fixed_x_indices,fixed_x_right, upper_or_lower_vec_right, l̄))
        if pruning_enable
            check_lb_pruning(right_node, best_ub)
        end
        # only perform upper bound calculation if not pruned:
        if ~right_node.data.is_pruned
            ū, feasible_x_right = compute_ub(right_solver, n,integer_vars,relaxed_x_right, luS,debug_print)
            # println("Right node, solved for ū: ", ū)
            best_ub, best_feasible_solution, fea_iter, fea_time, fea_fact_time, fea_solve_time, fea_nodes = update_ub(ū, feasible_x_right, best_ub, best_feasible_solution, right_node.data.depth, total_iter, fea_iter, total_time, fea_time, total_fact_time, fea_fact_time, total_solve_time, fea_solve_time, total_nodes,fea_nodes)
            push!(node_queue,right_node)
        end
        if debug_print
            println("fixed indices on right branch are : ", fixed_x_indices, " to ", fixed_x_right)        
        end
        ind = 1
        while ind ≤ lastindex(node_queue)
            if check_lb_pruning(node_queue[ind],best_ub)
                # printstyled("Fathom node in queue with lb > U!\n", color = :red)
                deleteat!(node_queue,ind)
            end
            ind += 1
        end
        
        iteration += 1 #TODO: could increment by 2 to count number of solved (or early_terminated) QPs
        # println("BnB loop iteration : ", iteration)
    end
    
    return best_ub, best_feasible_solution, early_num, total_iter, fea_iter, total_time, fea_time, total_fact_time, fea_fact_time, total_solve_time, fea_solve_time, total_nodes,fea_nodes
end
