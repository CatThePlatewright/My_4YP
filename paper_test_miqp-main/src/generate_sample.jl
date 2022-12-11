using SparseArrays, LinearAlgebra, Random, StatsBase
using NPZ

function generate_random_sample(m, n, inum, sparsity, rank)
    Q, R = qr(randn(n, n)); #or randn(n,n)
    D = diagm(0 => ones(rank) + 10*rand(rank));
    P = Q[:,1:rank]*D*Q[:,1:rank]';

    # M = sprand(n, n, 0.1)
    # P = Matrix(M*M');
    # P .+= diagm(0 => ones(rank) + 10*rand(rank))

    q = randn(n);
    A = -sprandn(m, n, sparsity);
    u = 2*ones(m) + rand(m);
    l = -2*ones(m) + rand(m);
    # u = 0.5*n*(ones(m) + rand(m));
    # l = 0.5*n*(-ones(m) + rand(m));
    lb = -2*ones(n) + rand(n);
    ub = 2*ones(n) + rand(n);
    b = zeros(m);
    #ensure integer feasibility
    # x0 = 2*randn(n) .-1.0;
    # @. x0 = round(x0)
    # b = A*x0;
    # println("x0 is ", x0)


    index_set =sample(1:n, inum, replace = false);
    # @. lb[index_set] = round(lb[index_set])
    # @. ub[index_set] = round(ub[index_set])

    #set as binary integer constraints
    tmp_1 = zeros(length(index_set))
    tmp_2 = ones(length(index_set))
    @. lb[index_set] = tmp_1
    @. ub[index_set] = tmp_2

    return P, q, A, b, l, u, lb, ub, index_set
end

"""
min 0.5 x'Px + q'x
s.t.  l ≤ Ax ≤u,
	  x ∈ Z, relaxed to [lx, ux]
"""

function generate_MPC(index)
    adaptive_data = npzread("mpc_data\\N=8\\adaptive_data.npy")
    fixed_data = npzread("mpc_data\\N=8\\matrice_data.npy")
    P = fixed_data["P"]
    q = adaptive_data["q_array"][index,:]
    A = -fixed_data["A"]
    b = zeros(size(A,1))
    println(cond(P))
    l = b + fixed_data["l"]
    u = b + adaptive_data["q_u_array"][index,:]
    lb = fixed_data["i_l"]
    ub = fixed_data["i_u"]
    index_set = fixed_data["i_idx"] .+ 1
    return sparse(P), q, sparse(A), b, l, u, lb, ub, index_set
end