using LinearAlgebra, SparseArrays, IterativeSolvers
import Base.size, Base.print
using Revise

includet("types.jl")


# """
# 	ADMMOperator

# ADMM in nonexpansive operator form `v_{k+1} = T(v_k)`. Solves QPs with interval constraints:
# min 0.5 x'Px + q'x
# s.t.  Ax + s == b,
# 	  s ∈ S
# """
# mutable struct ADMMOperator
# 	# problem data
# 	P::AbstractMatrix
# 	q::AbstractVector
# 	A::AbstractMatrix
# 	b::AbstractVector
# 	constraints::Array
# 	# parameters
# 	sigma::Float64   # step size
# 	ρ::Float64	 # step size
# 	alpha::Float64	 # over-relaxation
# 	max_iter::Int64  # maximum number of iterations
# 	iter::Int64      # current iteration counter
# 	eps_abs::Float64 # absolute tolerance
# 	eps_rel::Float64 # relative tolerance
# 	# KKT factorisation
# 	# fact             # stored factorisation
# 	indirect_mat::AbstractMatrix
# 	r1::AbstractVector
# 	r2::AbstractVector
# 	# Workspace
# 	x::AbstractVector # primal variable
# 	y::AbstractVector # dual variable
# 	s::AbstractVector # slack variable
# 	xt::AbstractVector
# 	st::AbstractVector
# 	ν::AbstractVector # dual of inner eq-QP
# 	rhs::AbstractVector
# 	sol::AbstractVector
# 	#residuals
# 	r_prim::Float64
# 	r_dual::Float64
# 	status::Symbol # solution status
# 	pre_dual_cost::Any

# 	function ADMMOperator(P::AbstractMatrix{T}, q::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, constraints, sigma::T, ρ::T, alpha::T, max_iter::Int64, eps_abs::T, eps_rel::T) where {T <: AbstractFloat}
# 		# determine problem dimension
# 		# define A = [-I; A_true] where the first n rows are used for branch and bound
# 		m, n = size(A);
# 		# # assemble KKT matrix
# 		# KKT = [P+Diagonal(sigma * ones(n)) A'; A diagm(0 => -1/ρ * ones(m))]
# 		# # factor KKT matrix and store factorisation
# 		# fact = ldlt(KKT)
# 		indirect_mat = P + diagm(0 => sigma * ones(n)) + ρ * A' * A
# 		r1 = zeros(n)
# 		r2 = zeros(m)
# 		# create workspace
# 		x = zeros(n)
# 		y = zeros(m)
# 		s = zeros(m)
# 		xt = zeros(n)
# 		st = zeros(m)
# 		ν = zeros(m)
# 		rhs = zeros(n)
# 		sol = zeros(m + n)
# 		iter = 0
# 		r_prim = 0.
# 		r_dual = 0.
# 		status = :unsolved
# 		pre_dual_cost = -Inf
# 		new(P, q, A, b, constraints, sigma, ρ, alpha, max_iter, iter, eps_abs, eps_rel, indirect_mat, r1, r2, x, y, s, xt, st, ν, rhs, sol, r_prim, r_dual, status, pre_dual_cost)
# 	end
# end

# size(op::ADMMOperator) = size(op.A)

"""
	MIADMMOperator

ADMM in nonexpansive operator form `v_{k+1} = T(v_k)`. Solves QPs with interval constraints:
min 0.5 x'Px + q'x
s.t.  Ax + s == b,
	  x ∈ Z, relaxed to [lx, ux]
	  adding another n constraints
rewritten as
s.t.  [-I; A]x + [sx; s] == [0; b]
		lx <= sx <= ux
		s ∈ S

Dual:
max  0.5 x'Px + b'y - σ_{S}(y) - σ_{[lx, ux]}(yx)
s.t.	Px - A'y + yx == -q
		y ∈ (K^{∞})^∘
		yx ∈ R^n
"""

mutable struct MIADMMOperator
	# problem data
	P::AbstractMatrix
	q::AbstractVector
	A::AbstractMatrix
	b::AbstractVector
	constraints
	lb::AbstractVector	#boundedness constraints
	ub::AbstractVector
	i_idx::AbstractVector	#integer index
	#scaling Information
	sm::ScaleMatrices
	# parameters
	sigma::Float64   # step size
	ρ::AbstractVector	 # step size
	ρ_x::AbstractVector   #step size
	alpha::Float64	 # over-relaxation
	max_iter::Int64  # maximum number of iterations
	iter::Int64      # current iteration counter
	eps_abs::Float64 # absolute tolerance
	eps_rel::Float64 # relative tolerance
	# KKT factorisation
	# fact             # stored factorisation
	indirect_mat::AbstractMatrix
	r1::AbstractVector
	r2::AbstractVector
	# Workspace
	x::AbstractVector # primal variable
	y::AbstractVector # dual variable
	yx::AbstractVector #dual variable for x ∈ [lx, ux]
	s::AbstractVector # slack variable
	sx::AbstractVector # slack variable for x ∈ [lx, ux]
	xt::AbstractVector
	st::AbstractVector
	ν::AbstractVector # dual of inner eq-QP
	rhs::AbstractVector
	sol::AbstractVector
	#residuals
	r_prim::Float64
	r_dual::Float64
	status::Symbol # solution status
	pre_dual_cost::Any
	δy::AbstractVector
	δyx::AbstractVector

	function MIADMMOperator(P::AbstractMatrix{T}, q::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, constraints, lb, ub, i_idx::AbstractVector, sigma::T, ρ::AbstractVector{T}, ρ_x::AbstractVector{T}, alpha::T, max_iter::Int64, eps_abs::T, eps_rel::T) where {T <: AbstractFloat}
		# determine problem dimension
		# define A = [-I; A_true] where the first n rows are used for branch and bound
		m, n = size(A);

		sm = ScaleMatrices(m, n)
		# # assemble KKT matrix
		# KKT = [P+Diagonal(sigma * ones(n)) A'; A diagm(0 => -1/ρ * ones(m))]
		# # factor KKT matrix and store factorisation
		# fact = ldlt(KKT)
		indirect_mat = P + diagm(0 => sigma * ones(n)) + diagm(0 => ρ_x) + A'* diagm(0 => ρ) * A	#compared to ADMMOperator, add a new term  ρ_x*I
		r1 = zeros(n)
		r2 = zeros(m)
		# create workspace
		x = zeros(n)
		y = zeros(m)
		yx = zeros(n)
		s = zeros(m)
		sx = zeros(n)
		xt = zeros(n)
		st = zeros(m)
		ν = zeros(m)
		rhs = zeros(n)
		sol = zeros(m + n)
		iter = 0
		r_prim = 0.
		r_dual = 0.
		status = :unsolved
		pre_dual_cost = -Inf
		δy = similar(y)
		δyx = similar(yx)

		#bounding l ≤ AX ≤ u via lb ≤ x ≤ ub
		if constraints[1] == :qp
			new_constraints = deepcopy(constraints)
			for i = 1:m
				ll = 0.0
				uu = 0.0

				column, value = findnz(A[i,:])
				for j in column
					if A[i,j] < 0
						ll += A[i,j]*ub[j]
						uu += A[i,j]*lb[j]
					else
						ll += A[i,j]*lb[j]
						uu += A[i,j]*ub[j]
					end
				end
				new_constraints[2][i] = max(constraints[2][i], ll)
				new_constraints[3][i] = min(constraints[3][i], uu)
			end
		end
		#change form according to Ax + s = b

		new(P, q, A, b, new_constraints, lb, ub, i_idx, sm, sigma, ρ, ρ_x, alpha, max_iter, iter, eps_abs, eps_rel, indirect_mat, r1, r2, x, y, yx, s, sx, xt, st, ν, rhs, sol, r_prim, r_dual, status, pre_dual_cost, δy, δyx)
	end
end

size(op::MIADMMOperator) = size(op.A)

function proj_box!(s::AbstractVector{Float64}, l::AbstractVector{Float64}, u::AbstractVector{Float64})
	#Or use clamp!()
	@. s = max(s, l)
    @. s = min(s, u)
end
function proj_box!(s::AbstractVector{Float64}, l::AbstractVector{Int64}, u::AbstractVector{Int64})
	@. s = max(s, l)
    @. s = min(s, u)
end

#projection onto sdp cone
function proj_sdp!(s::AbstractVector{Float64}, n)
	mat = reshape(s,n,n)
	obj = LinearAlgebra.eigen(mat)
	nonneg_eigv = zeros(n)
	# println(" # of neg eigenvalues: ", count(i->(i < 0), obj.values))
	for i = 1:n
		nonneg_eigv[i] = max(obj.values[i], 0)
	end
	mat = obj.vectors*diagm(nonneg_eigv)*obj.vectors'
	@. s = mat[:]
end

# function next!(v::AbstractVector, op::ADMMOperator)
# 	op.iter += 1
# 	m, n = size(op)

# 	# project first
# 	@. op.x = v[1:n]
# 	# projection
# 	@. op.s = v[n+1:n+m]
# 	if !isempty(op.s)
# 		if op.constraints[1] == :qp
# 			proj_box!(op.s, op.constraint[2], op.constraint[3])
# 		elseif op.constraints[1] == :sdp
# 			proj_sdp!(op.s, Int(sqrt(m)))
# 		end
# 	end
# 	@. op.y = op.ρ *(v[n+1:n+m] - op.s)

# 	# compute the residuals and check termination
# 	r_prim, r_dual = compute_residuals(op, op.x, op.s, op.y)

# 	op.r_prim = r_prim
# 	op.r_dual = r_dual
# 	has_converged(op, r_prim, r_dual) && return nothing

# 	# Indirect method
# 	@. op.r1 = op.sigma * 2 * op.x  - op.sigma * v[1:n] - op.q
# 	@. op.r2 = op.b - 2 * op.s + v[n+1:n+m]
# 	op.rhs = op.r1 + op.ρ * op.A'*op.r2
# 	#CG method for the indirect linear system
# 	IterativeSolvers.cg!(op.xt, op.indirect_mat, op.rhs)
# 	op.ν =  op.A*op.xt-op.r2
# 	op.sol[1:n] = op.xt
# 	@. op.sol[n+1:n+m] = op.ρ * op.ν
# 	@. op.st = 2 * op.s - v[n+1:n+m] - op.ν

# 	# over-relaxation
# 	@. v[1:n] = v[1:n] + op.alpha * ( op.xt - op.x) #here, op.x is indeed v[1:n]
# 	@. v[n+1:n+m] = v[n+1:n+m] +  op.alpha * ( op.st - op.s) #here, op.s is indeed v[n+1:n+m]

# 	op.iter == op.max_iter && (op.status = :max_iter_reached)

# 	return nothing
# end

function next!(
	v::AbstractVector{T}, 
	op::MIADMMOperator
) where {T}
	op.iter += 1
	m, n = size(op)

	# project first
	@. op.x = v[1:n]
	# projection
	@. op.sx = v[n+1:n+n]
	@. op.s = v[n+n+1:n+n+m]

	if !isempty(op.s)
		#scaling integer constraints
		if op.constraints[1] == :qp
			proj_box!(op.s, op.constraints[2], op.constraints[3])
			proj_box!(op.sx, op.sm.Dinv.diag.*op.lb, op.sm.Dinv.diag.*op.ub)	#project x onto the box constraint
		elseif op.constraints[1] == :sdp
			proj_sdp!(op.s, Int(sqrt(m)))
			proj_box!(op.sx, op.sm.Dinv.diag.*op.lb, op.sm.Dinv.diag.*op.ub)	#project x onto the box constraint
		end
	end

	#Update y, yx and δy, δyx
	@. op.δy = - op.y
	@. op.δyx = - op.yx
	@. op.y = op.ρ *(v[n+n+1:n+n+m] - op.s)
	@. op.yx = op.ρ_x*(v[n+1:n+n] - op.sx)
	@. op.δy = op.δy + op.y
	@. op.δyx = op.δyx + op.yx

	# compute the residuals and check termination
	compute_residuals(op)

	# println("primal res: ", op.r_prim, "      dual_res: ", op.r_dual)

	has_converged(op) && return nothing

	#Infeasibility detection
	if op.iter% 25 == 0
		infeasibility_detection(op) && return nothing
	end
	if op.iter% 100 == 0
		# infeasibility_detection(op) && return nothing
		# adaptive_step(op)
	end

	# Indirect method
	@. op.r1 = op.sigma * op.x - op.q + op.ρ_x * op.sx - op.yx
	@. op.r2 = op.b - 2 * op.s + v[n+n+1:n+n+m]		#Equivalent to op.b - op.s + op.y/op.ρ
	# op.rhs = op.r1 +  op.A'*diagm(0 => op.ρ)*op.r2
	mul!(op.rhs, op.A'*diagm(0 => op.ρ), op.r2)
	@. op.rhs = op.r1 + op.rhs

	#CG method for the indirect linear system
	IterativeSolvers.cg!(op.xt, op.indirect_mat, op.rhs)
	# op.ν =  op.A*op.xt-op.r2	#not producred by op.ρ for later use
	mul!(op.ν, op.A, op.xt)
	@. op.ν = op.ν - op.r2
	# op.sol[1:n] = op.xt
	# @. op.sol[n+1:n+m] = op.ρ * op.ν
	@. op.st = 2 * op.s - v[n+n+1:n+n+m] - op.ν		#Equivalent to op.s - op.ν + op.y/op.ρ; op.ν is not producted by op.ρ before and thus doesn't need to be normalized

	# over-relaxation
	@. v[1:n] = v[1:n] + op.alpha * ( op.xt - op.x) #here, op.x is indeed v[1:n]
	@. v[n+1:n+n] = v[n+1:n+n] + op.alpha * (op.xt - op.sx) #here, op.sxt is indeed op.xt
	@. v[n+n+1:n+n+m] = v[n+n+1:n+n+m] +  op.alpha * ( op.st - op.s) #here, op.s is indeed v[n+1:n+m]

	op.iter == op.max_iter && (op.status = :max_iter_reached)

	return nothing
end

#Currently, we only care about primal infeasibility since x is bounded and hence the objective function is lower-bounded
function infeasibility_detection(op::MIADMMOperator)
	D = op.sm.D
	E = op.sm.E
	Dinv = op.sm.Dinv
	Einv = op.sm.Einv

	#unscaled version
	residual = norm(Dinv.diag .*((op.A'*op.δy) - (op.δyx)), Inf)/norm([E.diag .*op.δy; Dinv.diag .*op.δyx], Inf)
	if op.constraints[1] == :qp
		obj_val = support_box!(op.δy, op.constraints[2], op.constraints[3]) + support_box!(Dinv.diag .*op.δyx, op.lb, op.ub) - op.b'*op.δy
		obj_val = obj_val/norm([E.diag .*op.δy; Dinv.diag .*op.δyx], Inf)
	end

	# println("residual: ", residual, "   obj_val: ", obj_val)
	#check primal infeasibility
	if residual < op.eps_rel && obj_val < op.eps_rel
		op.status = :primal_infeasible
		println("Detect primal infeasibility")
		return true
	else
		return false
	end
end

#adaptive step
function adaptive_step(op::MIADMMOperator)
    #adaptive_ρ, ρx, Same as OSQP
	mag_prim = max(max(norm([op.A*op.x; op.x], Inf), norm([op.s; op.sx], Inf)), norm(op.b, Inf))
	mag_dual = max(max(norm(op.P*op.x, Inf), norm(op.A'*op.y, Inf)), max(norm(op.yx, Inf), norm(op.q, Inf)))
	#No dependence on sx, yx
	mag_prim = max(max(norm(op.A*op.x, Inf), norm(op.s, Inf)), norm(op.b, Inf))
	mag_dual = maximum([norm(op.P*op.x, Inf), norm(op.A'*op.y, Inf), norm(op.q, Inf)])

	ratio = sqrt((op.r_prim/(mag_prim + 1e-10))/(op.r_dual/(mag_dual + 1e-10)))
	# println("ratio is: ", ratio)
	pre_ρ = deepcopy(op.ρ)
	pre_ρ_x = deepcopy(op.ρ_x)
	@. op.ρ_x = max(min(op.ρ_x*ratio, 1e5), 1e-5)
	@. op.ρ = max(min(op.ρ*ratio, 1e5), 1e-5)

	# @. op.ρ_x = op.ρ_x*ratio
	# @. op.ρ = op.ρ*ratio
	# if (norm([op.ρ; op.ρ_x], Inf) > 1e8)
	# 	error("ρ is too large")
	# end

	#The factorized matrix should also be modified
	n = length(op.x)
	op.indirect_mat .+= diagm(0 => (op.ρ_x - pre_ρ_x)) + op.A' * diagm(0 => (op.ρ - pre_ρ)) * op.A


    # mul = 10.0
    # ratio = op.r_prim/op.r_dual
    # if ratio > 1e4
    #     op.ρ_x = op.ρ_x*mul
    #     op.ρ = op.ρ*mul
    # elseif ratio < 1e-4
    #     op.ρ_x = op.ρ_x/mul
    #     op.ρ = op.ρ/mul
    # else
    #     op.ρ_x = op.ρ_x
    #     op.ρ = op.ρ
    # end
    # println("Iteration: ", operator.iter, "  ", norm(operator.A*operator.x, Inf), "  ", norm(operator.s, Inf), "  ", norm(operator.x, Inf), "  ", norm(operator.sx, Inf))
    # println("Iteration: ", operator.iter, "  primal norm is ", mag_prim, "  dual norm is ", mag_dual)
    # println("Iteration: ", operator.iter, "  Primal residual is ", operator.r_prim, "  Dual residual is ", operator.r_dual)
end

#= function compute_residuals(op::ADMMOperator, x, s, y)
	A= op.A
	b= op.b
	P= op.P
	q = op.q
	r_prim = norm(A * x + s - b, Inf)
	r_dual = norm(P * x + q - A' * y, Inf)

	return r_prim, r_dual
end =#

#scaled
function compute_residuals(op::MIADMMOperator)
	D = op.sm.D
	Dinv = op.sm.Dinv
	Einv = op.sm.Einv
	cinv = op.sm.cinv

	op.r_prim = norm([(Einv.diag).*(op.A * op.x + op.s - op.b); (D.diag).*(op.x - op.sx)], Inf)
	op.r_dual = norm(cinv*(Dinv.diag).*(op.P * op.x + op.q - op.A' * op.y + op.yx), Inf)
end

function has_converged(op)
	if op.r_prim <= op.eps_abs && op.r_dual <= op.eps_abs
		op.status = :solved
		return true
	else
		return false
	end
end

#unscaled
function extract_solution(op)
	x = op.x
	cost = 1/2 * x' * op.P * x + op.q' * x
	return op.x, op.s, op.y, cost
end

#support function
function support_box!(y, l, u)
    cost = 0
    for i = 1:length(y)
        cost = cost + max(y[i], 0)*u[i] + min(y[i], 0)*l[i]
    end
    return cost
end

print(op) = println(">>> ADMM-Iteration terminated!\n    Status: $(op.status)\n    Iter: $(op.iter)\n")
