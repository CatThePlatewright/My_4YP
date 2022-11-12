using Test
include("brute_recursion.jl")
include("mixed_binary_solver.jl")

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])
function simple_QP_params(Type::Type{T}) where {T <: AbstractFloat} 
    optimizer = Gurobi.Optimizer
    n = 5
    k = 3
    Q = Matrix{T}(I, n, n) #try sth else
    #Q = rand(n,n)
    #Q = Q'*Q
    Random.seed!(1234)
    c = rand(T,n)
    ϵ = 0.00000001
    #= P = spzeros(T,3,3)
    A = sparse(I(3)*T(1.))
    A = [A;-A].*2
    c = T[3.;-2.;1.]
    b = ones(T,6)
    cones = [Clarabel.NonnegativeConeT(3), Clarabel.NonnegativeConeT(3)]
 =#
    return (optimizer, n, k, Q,c,ϵ)
end


@testset "Basic Tests" begin
    for FloatT in UnitTestFloats
        @testset "Basic QP Tests (T = $(FloatT))" begin

            tol = FloatT(1e-3)
            @testset "feasible_binary" begin

                optimizer, n, k, Q,c,ϵ = simple_QP_params(FloatT)
                base_model = build_unbounded_base_model(optimizer,n,k,Q,c)
                (root,status) = branch_and_bound_solve(base_model,optimizer,n,ϵ)
                # put a termination status check function in your bnb solver, 
                # checking if lb=ub
                println("HEERE: ",typeof(status))
                @test status  == "OPTIMAL"

                # check against binary solver in Gurobi
                bin_model = Model(optimizer)
                set_optimizer_attribute(bin_model, "OutputFlag", 0)
                binary_model = build_base_model(bin_model,n,k,Q,c,collect(1:n))
                optimize!(binary_model)

                @test isapprox(norm(root.data.solution_x - FloatT.(value.(binary_model[:x]))), zero(FloatT), atol=tol)
                @test isapprox(root.data.ub, FloatT(objective_value(binary_model)), atol=tol)

            end
            @testset "feasible_mixed_binary" begin
                optimizer = Gurobi.Optimizer
                n = 8
                k= 5
                m = 4 # how many binary variables
                Q = Matrix{FloatT}(I, n, n) 
                Random.seed!(1234)
                c = rand(FloatT,n)
                ϵ = 0.00000001

                base_model = build_unbounded_base_model(optimizer,n,k,Q,c)
                binary_vars = sample(1:n, m, replace = false)
                sort!(binary_vars)
                println(binary_vars)
                root,term_status = branch_and_bound_solve(base_model,optimizer,n,ϵ, binary_vars)
                @test term_status == "OPTIMAL"
                println("Found objective: ", root.data.ub, " using ", root.data.solution_x)

                # check against binary solver in Gurobi
                bin_model = Model(optimizer)
                set_optimizer_attribute(bin_model, "OutputFlag", 0)
                binary_model = build_base_model(bin_model,n,k,Q,c,binary_vars)
                optimize!(binary_model)
                println("Exact solution: ", objective_value(binary_model) , " using ", value.(binary_model[:x]))
                println(" ")
                println(" ")
                @test isapprox(norm(root.data.solution_x - FloatT.(value.(binary_model[:x]))), zero(FloatT), atol=tol)
                @test isapprox(root.data.ub, FloatT(objective_value(binary_model)), atol=tol)
            end

            @testset "primal infeasible" begin
                println("Starting Primal Infeasible Test")
                optimizer = Gurobi.Optimizer
                n = 2
                k= 1
                Q = Matrix{FloatT}(I, n, n) 
                Random.seed!(1234)
                c = rand(FloatT,n)
                ϵ = 0.00000001
                
                base_model_infeasible = build_unbounded_base_model(optimizer,n,k,Q,c)
                x = base_model_infeasible[:x]
                #adding linear constraints to form an binary infeasible trapeze
                @constraint(base_model_infeasible, c1, x[1] + x[2] >= 0.5) 
                @constraint(base_model_infeasible, c2, - x[1] + x[2] >= -0.5) 
                @constraint(base_model_infeasible, c3, - x[1] + x[2] <= 0.5) 
                @constraint(base_model_infeasible, c4, x[1] + x[2] <= 1.5) 
                root,term_status = branch_and_bound_solve(base_model_infeasible,optimizer,n,ϵ)
                print(root.data.model)
# TODO: unbounded case??? i.e. dual infeasible (= primal unbounded)
                
                @test term_status  == "INFEASIBLE"

            end

    #=      @testset "primal infeasible" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                b[1] = -1
                b[4] = -1

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.PRIMAL_INFEASIBLE

            end

            @testset "dual infeasible" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                A[4,1] = 1.  #swap lower bound on first variable to redundant upper bound
                c .= FloatT[1.;0;0]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE

            end

            @testset "dual infeasible (ill conditioned)" begin

                P,c,A,b,cones = basic_LP_data(FloatT)
                A[1,1] = eps(FloatT)
                A[4,1] = -eps(FloatT)
                c .= FloatT[1.;0;0]

                solver   = Clarabel.Solver(P,c,A,b,cones)
                Clarabel.solve!(solver)

                @test solver.solution.status == Clarabel.DUAL_INFEASIBLE

            end =#

        end      #end "Basic LP Tests (FloatT)"
    end
end # UnitTestFloats
