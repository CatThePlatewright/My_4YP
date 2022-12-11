using LinearAlgebra, SparseArrays, Random, Test
using Debugger
using Revise

# you can do include of the code here
# include("./admm_operator.jl")
# ...but I suggest using the Revise package and using includet, that
# reloads any code changes automatically
#includet("../src/admm_operator.jl")
includet("../src\\admm_operator.jl")
includet("../src\\branch_and_bound.jl")
includet("../src\\cuts.jl")


# let's define a test QP
# min 	0.5 x' [4. 1; 1 2] x + [1; 1]' x
# s.t.  [1; 0; 0] <= [1  1; 1 0; 0 1] x + zeros(3) <= [1; 0.7; 0.7]
# function qp()
q = [1; 1.; 2.];
P = [2. -1. -1.; -1. 3. 1.; -1. 1. 2.];
A = -[2. 1. 1.];
l = -[-1.];
u = [3.];
m, n = size(A);
li = -3*ones(n);	#relaxed box constraints for integer constraints
ui = 3*ones(n);		#use datatype Float64 for simplicity
b = zeros(m);
#If init the problem without constraints
# A_init =  Array{Float64}(undef,0,length(q))
# b_init = Vector{Float64}(undef,0)
# l = Vector{Float64}(undef,0)
# u = Vector{Float64}(undef,0)


# set up operator
sigma = 1e-6;
rho = 0.1;
alpha = 1.6;
max_iter = 1000;
eps_abs = 1e-4;
eps_rel = 1e-3; #this doesnt do anything atm
eps_int = 1e-3; #tolerance for integer check
max_cut = 2;
settings = Dict([("eps_int",eps_int)])
constraints = [:qp, l, u, li, ui]
operator = MIADMMOperator(P, q, A, b, constraints, sigma, rho, alpha, max_iter, eps_abs, eps_rel);
LR_operator = OuterMIQP(settings, P, q, A, b, l, u, li, ui, sigma, rho, alpha, max_iter, eps_abs, eps_rel);

# # initialise ADMM operator variable v
v = zeros(n + m + n);
leaf = []
while !isempty(LR_operator.leaves)
	leaf = select_leaf(LR_operator)

	#solve the relaxed QP problem at leaf
	# start_time = time();
	solve_relaxed!(v, leaf)
	# end_time = time()

	#branch and bounds
	branch_and_bound(leaf, LR_operator)

	println(LR_operator.upper)
	#check whether terminate early
	if (LR_operator.upper - LR_operator.lower) < 1e-3
		break
	end

	LR_operator.iter_num += 1;
end

if isempty(LR_operator.leaves)
	println("We find the global optimum!")
	println("The total iteration number is: ", LR_operator.iter_num)
else
	println("We find a good feasible solution and terminate earlier!")
end

# #test outer approximation
# outer_index = 1
# max_outer = 10
# #Initialize v
# v = zeros(n+length(LR_operator.operator.b))
# out_x = []
# out_s = []
# out_y = []
# out_obj_val = Inf
# while outer_index < max_outer
# 	while true
# 		next!(v, LR_operator.operator);
# 		if LR_operator.operator.status != :unsolved
# 			print(LR_operator.operator)
# 			break
# 		end
# 	end
#
# 	#update lower & upper bound for outer approximation
# 	out_x, out_s, out_y, out_obj_val = extract_solution(v, LR_operator.operator)
# 	LR_operator.up_b = out_obj_val
# 	#reaches the admittable gap
# 	if abs(LR_operator.up_b - 1.88) < 1e-3
# 		break
# 	end
#
# 	#add cut
# 	# cut = hcat(A[outer_index,:], b[outer_index])
# 	add_cut!(LR_operator, A[outer_index,:], -b[outer_index])
# 	push!(LR_operator.operator.l, l[outer_index])
# 	push!(LR_operator.operator.u, u[outer_index])
#
# 	push!(LR_operator.operator.r2, 0.0)
# 	push!(LR_operator.operator.y, 0.0)
# 	push!(LR_operator.operator.ν, 0.0)
# 	push!(LR_operator.operator.st, 0.0)
# 	push!(LR_operator.operator.sol, 0.0)
# 	LR_operator.operator.iter = 0
# 	LR_operator.operator.status = :unsolved
# 	tmp = A[outer_index,:]
# 	#update the indirect linear system
# 	@. LR_operator.operator.indirect_mat += LR_operator.operator.rho*tmp'*tmp
# 	#Initialize v for the next iteration, as the dimension of s changes
# 	push!(v, 0.0)
# 	# v = zeros(length(v)+1)
#
# 	outer_index += 1
# end
#
# if outer_index == max_outer
# 	throw(OverflowError("The maximum iteration number reaches"))
# else
# 	println("the gap is ", LR_operator.up_b - 1.88)
# end

# # Let's check that the solution is correct:
# @testset "QP Problem" begin
#   @test norm(out_x - [0.3; 0.7], Inf) < 1e-3
#   @test abs(out_obj_val - 1.88) < 1e-3
# end
