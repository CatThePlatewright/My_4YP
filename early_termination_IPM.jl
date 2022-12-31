using SparseArrays, LinearAlgebra, Random, StatsBase
using NPZ
T = Float64
#include("direct_Clarabel_large_augmented.jl")
"""
	MIClarabel

IPM solves MIQP: 
min 0.5 x'Px + q'x
s.t.  Ax + s == b,
	  x ∈ Z, relaxed to [lx, ux]
      s ∈ ClarabelCones
rewritten as
s.t.  [A; I; -I]x + [s; s_; s+] == [b; u; -l]
		s ∈ S, 
        s_ and s+ are ClarabelNonnegativeCone

Dual:
max(wrt x, y, y(_+), y_) : -0.5 x'Px + b'*y + y(_+)'*u - y_'*l 
s.t.	Px + q - A'y - y_u + y_l == 0
		y ∈ C
		y_u and y_l <= 0
"""

"""early termination
- best_ub: current best upper bound on objective value stored in root node
- node: current node, needs to be implemented with Clarabel solver
"""
function early_termination(best_ub, node::BnbNode,m::Int)
    # check κ/τ before normalization
    ktratio = node.data.solver.info.ktratio
    if ktratio >= 1e-2
        node.data.is_pruned = true
        node.data.solver.status = Clarabel.DUAL_INFEASIBLE
        println("Infeasible ktratio, prune this node")
        return false
    end
    #iteration = node.data.solver.iteration # TOASK check if this is the wanted iteration?
    data = node.data.solver.data
    variables = node.data.solver.variables
    dual_cost = compute_dual_cost(data, variables,m) #compute current dual cost
    println("Found dual cost: ", dual_cost)
    # if (op.iter > 1) && (dual_cost < op.pre_dual_cost)
    #     println("Not monotonically increase at iteration ", op.iter, " with decrease ", dual_cost - op.pre_dual_cost)
    # end 

    #op.pre_dual_cost = dual_cost # TOASK: change this storage , e.g. to node.pre_dual_cost ???

    if (dual_cost > best_ub)
        println("early_termination has found dual_cost larger than best ub")
        node.data.solver.info.status = Clarabel.EARLY_TERMINATION
        node.data.is_pruned = true 
        # model.early_num += 1 TOASK: needed??
        return true
    end
    return false
end

"""
Dual cost computation for early termination
"""
#new framework for dual cost computation, 
# We can use qdldl.jl for optimization (17) later on.

function compute_dual_cost(data, variables::Clarabel.DefaultVariables{Float64}, m::Int) #Clarabeldata, Clarabelvariables
    τinv = inv(variables.τ)
    x = variables.x * τinv # normalize by τ
    y = - variables.z*τinv #include sign difference with Clarabel where z >= 0 but y_l and y_u are nonpositive
    println(" x :", x)
    println(" y : ",y)
    # correction by yminus and yplus (Method 2, auxiliary optimization)
    # yplus corresponds to s_(+) or lower bounds, yminus to s_(-) or upper bounds
    yplus = y[end-2*m+1:end-m] # last 2m rows of cones are the lower and upper bounds updated at each branching
    yminus = y[end-m+1:end]
    l = data.b[end-2*m+1:end-m]
    u = data.b[end-m+1:end]
    A0 = data.A[1:end-4*m, :] # TOASK I need to get original A and b back, any impact on other IPM calculations?
    b0 = data.b[1:end-4*m]
    y0 = y[1:end-4*m]
    println("A0 is ", A0)
    println("b0 is ", b0)

    n = length(data.q)
    Δx = zeros(n)

    # value of residual before the correction
    residual = data.P*x + data.q - A0'*y0 - yplus + yminus
    
    #dual correction, only for Δy = Δyplus - Δyminus
    Δy = residual 
    Δyplus = min(zeros(length(Δy)),Δy) 
    #mul!(Δx, op.P, coef*Δy) 
    cor_x = x + Δx # for simplicity, no correction for x

    #compute support function value S_{C}(y) of a box constraint 
    dual_cost = -0.5*cor_x'*data.P*cor_x + b0'*y0 + yplus'*u - yminus'*l + (Δyplus')*(u-l) + Δy'*l
    return dual_cost
end

function test_MIQP()
    n = 2
    m = 2
    k = 3
    P,q,A,b, cones, integer_vars= getData(n,m,k)
    
    Ā,b̄, s̄= getAugmentedData(A,b,cones,integer_vars,n)
    settings = Clarabel.Settings(verbose = true, equilibrate_enable = false, max_iter = 100)
    solver   = Clarabel.Solver()

    Clarabel.setup!(solver, P, q, Ā, b̄, s̄, settings)
    
    result = Clarabel.solve!(solver)
    # dual objective after correction
    dual_cost = compute_dual_cost(solver.data, solver.variables, m)
    # check with dual objective obtained from Clarabel when no early termination
    println("I found dual cost : ", dual_cost)
    println(" Clarabel info cost_dual ", solver.info.cost_dual)
end
"""
Penalized correction (ADMM paper)

min 0.5*α*||Δx||^2 + 0.5*γ*||Δy||^2
s.t.    P*Δx - Δy = -residual # NOTE: this is just my guess ?
where   residual := Px - A'*y - y + q
"""
function dual_correction!(data, Δx, Δy, residual)
    n = length(data.q)
    #coef = γ/α
    coef = 100
    #solve correction matrix, correction on y^k and x^k
    mat = coef*op.P*op.P + I
    IterativeSolvers.cg!(Δy, mat, -residual) # this solves (P^2*coef+I)*Δy = -r (eq.18)
    mul!(Δx, op.P, coef*Δy)
end


function generate_MPC(index)
    adaptive_data = npzread("mpc_data\\N=8\\adaptive_data.npy")
    fixed_data = npzread("mpc_data\\N=8\\matrice_data.npy")
    P = fixed_data["P"]
    q = adaptive_data["q_array"][index,:]
    A = -fixed_data["A"] # TOASK why -?
    b = zeros(size(A,1))
    println(cond(P))
    l = b + fixed_data["l"] #extended array of lower bounds
    # TOASK why not just l = fixed_data["l"]??
    u = b + adaptive_data["q_u_array"][index,:]
    lb = fixed_data["i_l"] #lower bound on integer variables
    ub = fixed_data["i_u"] # upper bound on integer variables
    index_set = fixed_data["i_idx"] .+ 1
    return sparse(P), q, sparse(A), b, l, u, lb, ub, index_set
end
