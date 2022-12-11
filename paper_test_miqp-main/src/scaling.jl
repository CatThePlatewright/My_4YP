using LinearAlgebra

include("algebra.jl")
includet("admm_operator.jl")

"""
Scaling matrices
"""

function kkt_col_norms!(P::AbstractMatrix{T}, A::AbstractMatrix{T}, norm_LHS::AbstractVector{T}, norm_RHS::AbstractVector{T}) where {T <: AbstractFloat}
	col_norms!(norm_LHS, P, reset = true);   #start from zero
	col_norms!(norm_LHS, A, reset = false);  #incrementally from P norms
	row_norms!(norm_RHS, A)                 #same as column norms of A'
	return nothing
end

function limit_scaling!(s::Vector{T}, MIN_SCALING, MAX_SCALING) where {T <: AbstractFloat}
	@.s = clip(s, MIN_SCALING, MAX_SCALING, one(T))
	return nothing
end

function limit_scaling(s::T, MIN_SCALING, MAX_SCALING) where {T <: AbstractFloat}
	s = clip(s, MIN_SCALING, MAX_SCALING, one(T))
	return s
end

function scale_ruiz!(op::MIADMMOperator)
	MIN_SCALING = 1e-4
	MAX_SCALING = 1e4
	m,n = size(op.A)

	#references to scaling matrices from operator
	D = op.sm.D
	E = op.sm.E
	c = op.sm.c

	#unit scaling to start
	D.diag .= ones(n)
	E.diag .= ones(m)
	c = 1.0

	#use the inverse scalings as intermediate
	#work vectors as well, since we don't
	#compute the inverse scaling until the
	#final step
	Dwork = op.sm.Dinv
	Ework = op.sm.Einv

	#references to QP data matrices
	P = op.P
	A = op.A
	q = op.q
	b = op.b

	#perform scaling operations for a fixed
	#number of steps, i.e. no tolerance or
	#convergence check
	for i = 1:100
		kkt_col_norms!(P, A, Dwork.diag, Ework.diag)
		limit_scaling!(Dwork.diag, MIN_SCALING, MAX_SCALING)
		limit_scaling!(Ework.diag, MIN_SCALING, MAX_SCALING)

		inv_sqrt!(Dwork.diag)
		inv_sqrt!(Ework.diag)

		# Scale the problem data and update the
		# equilibration matrices
		scale_data!(P, A, q, b, Dwork, Ework, 1.0)
		LinearAlgebra.lmul!(Dwork, D)        #D[:,:] = Dtemp*D
		LinearAlgebra.lmul!(Ework, E)        #D[:,:] = Dtemp*D

		# now use the Dwork array to hold the
		# column norms of the newly scaled P
		# so that we can compute the mean
		col_norms!(Dwork.diag, P)
		mean_col_norm_P = mean(Dwork.diag)
		inf_norm_q      = norm(q, Inf)

		if mean_col_norm_P  != 0. && inf_norm_q != 0.

			inf_norm_q = limit_scaling(inf_norm_q, MIN_SCALING, MAX_SCALING)
			scale_cost = max(inf_norm_q, mean_col_norm_P)
			scale_cost = limit_scaling(scale_cost, MIN_SCALING, MAX_SCALING)
			ctmp = 1.0 / scale_cost

			# scale the penalty terms and overall scaling
			scalarmul!(P, ctmp)
			q .*= ctmp
			c *= ctmp
		end
	end #end Ruiz scaling loop

    issymmetric(P) || symmetrize_full!(P)

	#update the inverse scaling data, c and c_inv
	op.sm.Dinv.diag .= 1.0 ./ D.diag
	op.sm.Einv.diag .= 1.0 ./ E.diag

    #scale set components
	scale_sets!(E, op)

    #These are Base.RefValue type so that
	#scaling can remain an immutable
	op.sm.c = c
	op.sm.cinv = 1.0 / c

	# scale the potentially warm started variables
	scale_variables!(op.x, op.y, op.yx, op.s, op.sx, op.sm.Dinv, op.sm.Einv, op.sm.E, op.sm.c)

	#update the indirect matrix
	op.indirect_mat .= op.P + diagm(0 => op.sigma * ones(n)) + diagm(0 => op.ρ_x) + op.A'* diagm(0 => op.ρ) * op.A
	return nothing
end

function scale_data!(P::AbstractMatrix{T}, A::AbstractMatrix{T}, q::AbstractVector{T}, b::AbstractVector{T}, Ds::AbstractMatrix{T}, Es::AbstractMatrix{T}, cs::T = one(T)) where {T <: AbstractFloat}

	lrmul!(Ds, P, Ds) # P[:,:] = Ds*P*Ds
	lrmul!(Es, A, Ds) # A[:,:] = Es*A*Ds
	q[:] = Ds * q
	b[:] = Es * b
	if cs != one(T)
		scalarmul!(P,cs)
		q .*= cs
	end
	return nothing
end

function inv_sqrt!(A::Vector{T}) where{T <: Real}
	@fastmath A .= one(T) ./ sqrt.(A)
end

function scale_sets!(E::AbstractMatrix, op::MIADMMOperator)
    if op.constraints[1] == :qp
        l = op.constraints[2]
        u = op.constraints[3]
        lmul!(E, l)
        lmul!(E, u)

        # keep integer constraints unchanged
        # lb = op.lb
        # ub = op.ub
        # lmul!(Dinv, lb)
        # lmul!(Dinv, ub)
    end
end

function scale_variables!(x, y, yx, s, sx, Dinv, Einv, E, c)
    x[:] = Dinv * x
	y[:] = Einv * y
    yx[:] = Dinv * yx
	s[:] = E * s
    sx[:] = Dinv * sx
	@. y *= c
    @. yx *= c
end

function reverse_scaling!(op::MIADMMOperator)

	cinv = ws.sm.cinv[] #from the Base.RefValue type
	ws.vars.x[:] = ws.sm.D * ws.vars.x
	ws.vars.s[:] = ws.sm.Einv * ws.vars.s
	ws.vars.μ[:] = ws.sm.E * ws.vars.μ
	ws.vars.μ  .*= cinv

	return nothing
end
