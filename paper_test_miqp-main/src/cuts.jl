using LinearAlgebra

#####################################################################################
# YC: This file is not related to early termination and no longer under development
#####################################################################################

#Add cutting planes for outer approximation
#length of "normal" is n
function add_cut!(outermodel::OuterMIQP, normal::Vector{T}, constant) where{T}
    # #test whether the preallocated memory of A is full
    # if (outermodel.num_cut < outermodel.max_cut)
    #     outermodel.num_cut += 1
    # else
    #     outermodel.max_cut += 10
    #     outermodel.num_cut += 1
    #     #add enough storage fpr new cuts
    #     outermodel.operator.A = vcat(outermodel.operator.A, Array{T,2}(undef, 10, length(normal)))
    #     outermodel.operator.b = vcat(outermodel.operator.b, Vector{T}(undef,10))
    #     outermodel.operator.s = vcat(outermodel.operator.s, Vector{T}(undef,10))
    # end

    #add new cut
    tmp = permutedims(normal)
    # outermodel.operator.A[outermodel.num_cut,:] = deepcopy(tmp)
    # outermodel.operator.b[outermodel.num_cut] = -constant   #need "-" sign here to transform ineq into standard eq
    # outermodel.operator.s[outermodel.num_cut] =  0.0
    outermodel.operator.A = vcat(outermodel.operator.A, tmp)
    outermodel.operator.b = vcat(outermodel.operator.b, -constant)
    outermodel.operator.s = vcat(outermodel.operator.s, 0.0)
    outermodel.operator.l = vcat(outermodel.operator.l, 0.0)
    outermodel.operator.u = vcat(outermodel.operator.u, Inf)
end



#---------------------------------------------
#                   Initial fixed cuts
#---------------------------------------------


#Exponential Cone


# Second-order Cone
function soc_init_cut!(soc_constraint::COSMO.Constraint{T}, LR_constraints::Vector{COSMO.Constraint{T}}) where{T}
    n = length(soc_constraint.b)
    A_init = zeros(T, 2*n-2, n)
    A_init[:,1] = ones(T, 2*n-2)
    A_init[1:n-1, 2:end] = Matrix{T}(I,n-1,n-1)
    A_init[n:2*n-2, 2:end] = -Matrix{T}(I,n-1,n-1)
    b_init = zeros(T, 2*n-2)
    init_constraint = COSMO.Constraint(A_init, b_init, COSMO.Nonnegatives)
    append!(LR_constraints, [init_constraint])
end


# SDP cone



#---------------------------------------------
#                   Certificate cuts
#---------------------------------------------



#---------------------------------------------
#                   Separation cuts
#---------------------------------------------
