#Branch and Bound
using Revise
using FileIO, JLD, JLD2 # to load the matrices from the file
includet("admm_operator.jl")
includet("scaling.jl")

#Information of a branching node
mutable struct Node
    lower       #lower bound at the node
    operator::MIADMMOperator    #combine data and solver in miosqp
    depth::Int64               #depth in the tree
    x::AbstractVector           #node's relaxed solution
    s::AbstractVector           #node's relaxed solution
    y::AbstractVector           #node's relaxed solution
    status
    # num_iter::Int64
    # nextvar_idx
    # constr_idx

    function Node(operator::MIADMMOperator, depth = 0, lower = -Inf)
        x = operator.x
        s = operator.s
        y = operator.y
        status = :unsolved
        # num_iter = 0
        # nextvar_idx = nothing
        # constr_idx = nothing
        return new(lower, operator, depth, x, s, y, status)
        # return new(lower, operator, depth, li, ui, x, s, y, status, num_iter, nextvar_idx, constr_idx)
    end
end


#List for branching nodes, plus cutting planes
########################################################
mutable struct OuterMIQP
    settings	   #settings for branch and bound
	operator
	leaves::AbstractVector     #leaves in the tree
	fea_x::AbstractVector      #current best solution
    fea_s::AbstractVector      #current best solution
	lower::AbstractFloat	   #lower bound for OuterMIQP
	upper::AbstractFloat	       #upper bound for OuterMIQP
    i_idx::AbstractVector       #index of integer variables in MIQP
    #test signal
    sign_early_termination::Bool
    iter_num::Int64 	#number for branch and bound
    early_num::Int64
    total_iter::Int64
    fea_iter::Any     #iteration that finds the first feasible solution
    fea_num::Any

	function OuterMIQP(settings, P, q, A, b, constraints, lb, ub, i_idx, sigma, ρ, ρ_x, alpha, max_iter, eps_abs, eps_rel, sign_early_termination)
		operator = MIADMMOperator(P, q, A, b, constraints, lb, ub, i_idx, sigma, ρ, ρ_x, alpha, max_iter, eps_abs, eps_rel)

        #preconditioning
        scale_ruiz!(operator)

        #Define the root node
        root = Node(operator)
        leaves = [root]				#leaves in the tree

        m, n = size(A)
        fea_x = zeros(n)	#current best solution
        fea_s = zeros(m)
        lower = -Inf
		upper = Inf

        iter_num = 0 	#number for branch and bound
        early_num = 0
        total_iter = 0
        fea_iter = NaN
        fea_num = NaN
		return new(settings, operator, leaves, fea_x, fea_s, lower, upper, i_idx, sign_early_termination, iter_num, early_num, total_iter, fea_iter, fea_num)
	end
end

#select a leaf from leaves for computing
function select_leaf(model::OuterMIQP)
    #depth first until find the first feasible solution
    if model.upper == Inf
        # println("Depth first")
        depth_set = []
        for elem in model.leaves
            push!(depth_set, elem.depth)
        end
        depth = maximum(depth_set)
        max_set = findall(x -> x == depth, depth_set)

        index = 1
        #Find the one with the lowest bound
        lower_bound = Inf
        for i = 1:length(max_set)
            if model.leaves[max_set[i]].lower < lower_bound
                index = i
            end
        end

        # index = argmax(x -> x.depth, model.leaves)
        return splice!(model.leaves, max_set[index])    #delete and return the selected leaf
    #best bound when we have a feasible solution
    else
        lower_set = []
        for elem in model.leaves
            push!(lower_set, elem.lower)
        end
        index = argmin(lower_set)
        return splice!(model.leaves, index)    #delete and return the selected leaf
    end
end

#solve relaxed QP
function solve_relaxed!(v::AbstractVector, node::Node, model::OuterMIQP)
    # to = TimerOutput()
    # @timeit to "domain propagation" 
    # print_timer(to)
    # reset_timer!()

    #Presolving
    if node.operator.constraints[1] == :qp
        domain_propagation(node.operator)
    end

    # if(model.iter_num == 3)
    #     println(model.iter_num)
    # end
    # tmp = zeros(length(v))
    # @. v = tmp
    # println(model.total_iter)

    # Main loop for ADMM like COSMO
    while true
        #iteration update
        next!(v, node.operator);
        model.total_iter = model.total_iter + 1

        #check early termination
        if (node.operator.iter % 25 == 0)
            # check early termination
            if model.sign_early_termination
                if model.upper < Inf
                    early_termination(model, node.operator)
                end
                # early_termination(model, node.operator)
            end

            # Status check
            if status_check(node,model)
                return
            end
        end
	end
    # #store the objective value of the relaxed QP at current node
    # node.x, node.s, node.y, node.lower = extract_solution(node.operator)
    return
end

function status_check(
    node::Node, 
    model::OuterMIQP
)
    if node.operator.status != :unsolved
        node.status = node.operator.status

        if node.status == :solved
            node.x, node.s, node.y, node.lower = extract_solution(node.operator)
            println("Node ", model.iter_num, " iteration ", node.operator.iter)
            # println("solved! with iteration ", node.operator.iter,"    Eq # ", sum(node.operator.lb .== node.operator.ub), "   rel lower ", (model.upper - model.lower)/abs(model.lower))
            # println("Node ", model.iter_num, "  x is ", node.operator.sm.D*node.x)
        elseif node.status == :early_termination
            println("Early termination at iteration", node.operator.iter)
        elseif node.status == :primal_infeasible
            println("Primal infeasible")
            println("Node ", model.iter_num, " iteration ", node.operator.iter)

            println("Primal infeasible with ", " Primal: ",norm(node.operator.A'*node.operator.δy - node.operator.δyx, Inf)/norm([node.operator.δy; node.operator.δyx], Inf), " with obj_val: ", (support_box!(node.operator.δy, node.operator.constraints[2], node.operator.constraints[3]) + support_box!(node.operator.δyx, node.operator.lb, node.operator.ub) - node.operator.b'*node.operator.δy)/norm([node.operator.δy; node.operator.δyx], Inf))
            # save("data\\problem.jld", "P", Matrix(node.operator.P), "q", node.operator.q, "A", Matrix(node.operator.A), "b", node.operator.b, "l", node.operator.constraints[2], "u", node.operator.constraints[3], "lb", node.operator.lb,
            #         "ub", node.operator.ub, "i_idx", node.operator.i_idx)
            # error("Primal infeasible ", model.iter_num)

        elseif node.status == :max_iter_reached
            println("maximum iteration reached with", " Primal: ", node.operator.r_prim, "Dual: ", node.operator.r_dual)
            println("Equality number", sum(node.operator.lb .== node.operator.ub))
            println("Infeasible primal: ",norm(node.operator.A'*node.operator.δy - node.operator.δyx, Inf)/norm([node.operator.δy; node.operator.δyx], Inf), " with obj_val: ", (support_box!(node.operator.δy, node.operator.constraints[2], node.operator.constraints[3]) + support_box!(node.operator.δyx, node.operator.lb, node.operator.ub) - node.operator.b'*node.operator.δy)/norm([node.operator.δy; node.operator.δyx], Inf))
            println("ρ value is ", node.operator.ρ, "  ", node.operator.ρ_x)
            # println("Primal residual is ", node.operator.r_prim, "  dual residual is ", node.operator.r_dual)

            # #store matrices after scale-ruiz, only lb,ub unscaled. Hence it is incomplete
            # save("data\\problem.jld", "P", Matrix(node.operator.P), "q", node.operator.q, "A", Matrix(node.operator.A), "b", node.operator.b, "l", node.operator.constraints[2], "u", node.operator.constraints[3], "lb", node.operator.lb,
            #         "ub", node.operator.ub, "i_idx", node.operator.i_idx)

            node.x, node.s, node.y, node.lower = extract_solution(node.operator)
            println("obj value: ", node.lower*node.operator.sm.cinv)
            println("Integer constraints:", node.operator.lb[node.operator.i_idx], node.operator.ub[node.operator.i_idx])
            error("Don't converge at node ", model.iter_num)
        end

        #count # of relaxed problem
        model.iter_num += 1;

        return true
    end

    return false
end

#branch and bound step
function branch_and_bound(node::Node, model::OuterMIQP)
    #1) Do B&B only when the problem is solved
    if node.status == :solved
        #2) If the objective value of the relaxed QP is greater than the current feasible solution, remove it
        if node.lower > model.upper
            println("Node's lower bound is still greater than the existing feasible solution!")
            return
        end

        #3)unscale variable
        D = node.operator.sm.D
        Einv = node.operator.sm.Einv
        cinv = node.operator.sm.cinv
        unscaled_x = D*node.x

        #Find indices of variable x with integer value
        tol = model.settings["eps_int"]
        int_idx = check_int_feasibility(unscaled_x[model.i_idx], tol)

        #3) If integer feasible,
        if sum(int_idx) == length(model.i_idx)
            # println("Find a feasible point")
            #compare with the current best solution
            if model.upper == Inf
                model.fea_iter = model.total_iter
                model.fea_num = model.iter_num
                println("Find the first one at iteration: ", model.total_iter, "corresponding to node ", model.iter_num)
            end
            if model.upper > node.lower
                model.upper = node.lower #store the scaled value, save computation
                model.fea_x = unscaled_x
                model.fea_s = Einv*node.s
                # println("A better feasible point")
            end

            #prune nodes
            if !isempty(model.leaves)
                filter!(ele -> (ele.lower < model.upper), model.leaves)
            end
            return
        end

        #heuristic guessing for fractional solution

        #branching
        # index = findfirst(.!int_idx)
        frac_set = setdiff(model.i_idx, model.i_idx[int_idx])
        # println("set difference: ", frac_set)
        index = pick_index(node, unscaled_x[frac_set])
        # println("Node ", model.iter_num-1, "   branching index ", frac_set[index])
        add_node(node, model, frac_set[index])
        # println("Integer index is: ", int_idx)
        # println("Current fractional set: ", frac_set)
        # println("The splitting index is: ", frac_set[index], " with value ", node.operator.x[frac_set[index]])
        # println("At the splitting node, lb is ", node.operator.lb[frac_set[index]], "ub is ", node.operator.ub[frac_set[index]])

        #update lower bound of the model
        if !isempty(model.leaves)
            model.lower = minimum(x -> x.lower, model.leaves)
        end
    end
end

#Note: The current pick_index() only suitable for fully integer QP rather than mixed integer QP. Need more modification for solving mixed integer QP
function pick_index(node::Node, x_int)
    frac_part = broadcast(v -> abs(v - round(v)), x_int)
    # println("pickindex frac_part: ", frac_part)
    max_val = maximum(frac_part)
    index = findfirst(v -> v == max_val, frac_part)
    # println("Possible branching index: ", findall(v -> v == max_val, frac_part))
    return index
end

function check_int_feasibility(sol_x, tol)
    #check integer feasibility entrywise
    int_idx = @. abs(sol_x - round(sol_x)) < tol

    return int_idx
end

function add_node(node::Node, model::OuterMIQP, index)
    #create new leaves

    #unscale the branching variable
    D = node.operator.sm.D
    unscaled_x = D.diag[index]*node.x[index]

    #As we use the first n columns of op.l for branching, it implicitly implies li.
    operator_l = deepcopy(node.operator);
    operator_l.ub[index] = floor(unscaled_x) #branching
    reset_inner_operator(operator_l)

    operator_r = deepcopy(node.operator);
    operator_r.lb[index] = ceil(unscaled_x)  #branching
    reset_inner_operator(operator_r)

    left_leaf = Node(operator_l, node.depth+1, node.lower)
    right_leaf = Node(operator_r, node.depth+1, node.lower)
    #add them to model.leaves
    push!(model.leaves, left_leaf)
    push!(model.leaves, right_leaf)
end

function reset_inner_operator(op::MIADMMOperator)
    op.status = :unsolved
    op.iter = 0
    m,n = size(op.A)

    #reset penalty parameter ρ
    op.ρ = 1e0 * ones(m);
    op.ρ_x = 1e0 * ones(n);

    #set ρ in equality constraints 1000x times larger
    if op.constraints[1] == :qp
        eq_ind = broadcast(==, op.constraints[2], op.constraints[3])
        @. op.ρ[eq_ind] = 1e3*op.ρ[eq_ind]

        eq_ind_x = broadcast(==, op.lb, op.ub)
        @. op.ρ_x[eq_ind_x] = 1e3*op.ρ_x[eq_ind_x]
    end

    #reset the indirect matrix due to the change of ρ
    op.indirect_mat .= op.P + diagm(0 => op.sigma * ones(n)) + diagm(0 => op.ρ_x) + op.A' * diagm(0 => op.ρ) * op.A
end


"""
Dual cost computation for early termination
"""
# #dual cost computation for SDP (old)
# function compute_dual_cost(op::ADMMOperator)
#     rhs = - op.q + op.A'*op.y
#     dual_x = op.P\rhs
#
#     #compute support function value S_{C}(y) of a box constraint
#     dual_cost = -0.5*dual_x'*rhs + op.b'*op.y
#
#     return dual_cost
# end
#new framework for dual cost computation
function compute_dual_cost(op::MIADMMOperator)
    #scaling factor
    Dinv = op.sm.Dinv

    #Method 1: correction by merely yx
    # cor_yx = -op.q - op.P*op.x + op.A'*op.y  # correction at yx by dual residual
    # cor_x = op.x

    #Method 2: correction by x, yx (auxiliary optimization)
    n = length(op.q)
    delta_x = zeros(n)
    delta_yx = zeros(n)
    residual = op.P*op.x - op.A'*op.y + op.yx + op.q
    dual_correction!(op, delta_x, delta_yx, residual)
    cor_x = op.x + delta_x
    cor_yx = op.yx + delta_yx

    #compute support function value S_{C}(y) of a box constraint
    if op.constraints[1] == :qp
        dual_cost = -0.5*cor_x'*op.P*cor_x + op.b'*op.y - support_box!(op.y, op.constraints[2], op.constraints[3]) - support_box!(cor_yx, Dinv*op.lb, Dinv*op.ub)
    elseif op.constraints[1] == :sdp
        dual_cost = -0.5*cor_x'*op.P*cor_x + op.b'*op.y - support_box!(cor_yx, Dinv*op.lb, Dinv*op.ub)
    end
    return dual_cost
end


"""
Penalized correction

min 0.5*α*||delta_x||^2 + 0.5*γ*||delta_yx||^2
s.t.    P*delta_x + delta_yx = -residual
where   residual := Px - A'*y + yx + q
"""
function dual_correction!(op::MIADMMOperator, delta_x, delta_yx, residual)
    n = length(op.q)
    #coef = γ/α
    coef = 100
    #solve correction matrix, correction on yx^k and x^k
    mat = coef*op.P*op.P + I
    IterativeSolvers.cg!(delta_yx, mat, -residual)
    mul!(delta_x, op.P, coef*delta_yx)
end


#early termination
function early_termination(model::OuterMIQP, op::MIADMMOperator)
    #y_k and ν_k are possible for dual generation
    upper = model.upper #existing feasible upper bound
    dual_cost = compute_dual_cost(op) #compute current dual cost
    # if (op.iter > 1) && (dual_cost < op.pre_dual_cost)
    #     println("Not monotonically increase at iteration ", op.iter, " with decrease ", dual_cost - op.pre_dual_cost)
    # end
    op.pre_dual_cost = dual_cost

    if (dual_cost > upper)
        # println("We terminate earlier before convergence: Iteration ", op.iter)
        op.status = :early_termination
        model.early_num += 1
    end
end

"""
Domain propagation (Presolving Techniques)
"""

function domain_propagation(op::MIADMMOperator)
    Dinv = op.sm.Dinv
    Einv = op.sm.Einv

    #unscaling matrices
    A = Einv * op.A * Dinv
    b = Einv * op.b
    #-Ax + s = b, s ∈ [l,u],shift back when doing domain propogation
    l = Einv*(op.constraints[2] - op.b)
    u = Einv*(op.constraints[3] - op.b)
    lb = op.lb
    ub = op.ub
    i_idx = op.i_idx

    m, n = size(A)
    flag = true #check whether domain propogation is effective

    while flag
        pre_lb = deepcopy(lb)
        pre_ub = deepcopy(ub)

        for row = 1:m
            #find nonzero items in each constraint
            column, value = findnz(A[row,:])

            for i = 1:length(column)
                upper = u[row]
                lower = l[row]
                for j = 1:length(column)
                    #except the index i
                    if i == j
                        continue
                    end
                    #check the sign of coefficient
                    if value[j] < 0 # as we store -A, value[j] < 0 means A[row,column[j]] > 0
                        upper -= -value[j]*lb[column[j]]
                        lower -= -value[j]*ub[column[j]]
                    else
                        upper -= -value[j]*ub[column[j]]
                        lower -= -value[j]*lb[column[j]]
                    end
                end

                #update bound
                if value[i] < 0
                    ub[column[i]] = min(ub[column[i]], -upper/value[i])
                    lb[column[i]] = max(lb[column[i]], -lower/value[i])
                else
                    ub[column[i]] = min(ub[column[i]], -lower/value[i])
                    lb[column[i]] = max(lb[column[i]], -upper/value[i])
                end

                #tighten bound for integer variables
                @. ub[i_idx] = floor(ub[i_idx])
                @. lb[i_idx] = ceil(lb[i_idx])
            end
        end

        #domain propogation ends
        if (lb == pre_lb) && (ub == pre_ub)
            flag = false
        end
        # else
        #     println("domain propogation works")
        # end
    end

    #only update lb ≤ x ≤ ub at present

    #detect infeasibility, return immediately
    if !(lb <= ub)
        op.status = :primal_infeasible
        println("Detect primal infeasibility in domain propogation")
    end

    #Finally, reset ρ_x when it is an equality constraint as lb,ub changed
    #reset penalty parameter ρ_x
    op.ρ_x = 1e0 * ones(n);

    #set ρ_x in equality constraints 1000x times larger
    eq_ind_x = broadcast(==, op.lb, op.ub)
    @. op.ρ_x[eq_ind_x] = 1e3*op.ρ_x[eq_ind_x]

    #reset the indirect matrix due to the change of ρ
    op.indirect_mat .= op.P + diagm(0 => op.sigma * ones(n)) + diagm(0 => op.ρ_x) + op.A' * diagm(0 => op.ρ) * op.A
end
