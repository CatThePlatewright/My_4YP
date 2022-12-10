
# -------------------------------------
# Problem scaling
# -------------------------------------

mutable struct ScaleMatrices
	D::Diagonal
	Dinv::Diagonal
	E::Diagonal
	Einv::Diagonal
	c::Float64
	cinv::Float64
end

ScaleMatrices(args...) = ScaleMatrices{Float64}(args...)

function ScaleMatrices(m, n)
	D    = Diagonal(ones(n))
	Dinv = Diagonal(ones(n))
	E    = Diagonal(ones(m))
	Einv = Diagonal(ones(m))
	c    = 1.0
	cinv = 1.0
	ScaleMatrices(D, Dinv, E, Einv, c, cinv)
end
