using Test
include("direct_Clarabel_one_solver.jl")

#if not run in full test setup, just do it for one float type
@isdefined(UnitTestFloats) || (UnitTestFloats = [Float64])
function simple_QP_params(Type::Type{T}) where {T <: AbstractFloat} 
    optimizer = Clarabel.Optimizer
    n = 15
    k = 26
    m=10
    integer_vars = sample(1:n, m, replace = false)
    sort!(integer_vars)
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
    return (optimizer, n, k, integer_vars, Q,c,ϵ)
end


@testset "Basic Tests" begin
    for FloatT in UnitTestFloats
        @testset "Basic QP Tests (T = $(FloatT))" begin

            tol = FloatT(1e-3)
            @testset "feasible_MILP_Clarabel" begin
                

                optimizer, n, k, integer_vars, Q,c,ϵ = simple_QP_params(FloatT)
                old_model = build_unbounded_base_model(optimizer,n,k,Q,c)
                solve_base_model(old_model,integer_vars)
                println("integer_vars: ", integer_vars)
                #solve in Clarabel the relaxed problem
                P,q,A,b, cones= getClarabelData(old_model)
                A,b, cones= getAugmentedData(A,b,cones,integer_vars,n)
                settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
                solver   = Clarabel.Solver()
                Clarabel.setup!(solver, P, q, A, b, cones, settings)
                result = Clarabel.solve!(solver)
                println("STARTING CLARABEL BNB LOOP ")
                root, term_status = branch_and_bound_solve(solver, result,n,ϵ, integer_vars)
                println("Found objective: ", root.data.ub, " using ", round.(root.data.solution_x,digits=3))

                @test term_status  == "OPTIMAL"

                # check against Gurobi
                # check against binary solver in Gurobi
                exact_model = Model(Gurobi.Optimizer)
                set_optimizer_attribute(exact_model, "OutputFlag", 0)
                x = @variable(exact_model, x[i = 1:n])
                for bin in integer_vars
                    set_integer(x[bin])
                end
                @objective(exact_model, Min, x'*Q*x + c'*x)
                @constraint(exact_model, sum_constraint, sum(x) == k)
                optimize!(exact_model)
                println("Exact solution: ", objective_value(exact_model) , " using ", value.(exact_model[:x])) 

                @test isapprox(norm(root.data.solution_x - FloatT.(value.(exact_model[:x]))), zero(FloatT), atol=tol)
                @test isapprox(root.data.ub, FloatT(objective_value(exact_model)), atol=tol)

            end

            @testset "primal infeasible" begin
                println("Starting Primal Infeasible Test")
                optimizer = Clarabel.Optimizer
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
                solve_base_model(base_model_infeasible, collect(1:n))

                #solve in Clarabel 
                P,q,A,b, cones= getClarabelData(base_model_infeasible)
                A,b, cones= getAugmentedData(A,b,cones,collect(1:n),n)
                settings = Clarabel.Settings(verbose = true, equilibrate_enable = false, max_iter = 100)
                solver   = Clarabel.Solver()
                Clarabel.setup!(solver, P, q, A, b, cones, settings)
                result = Clarabel.solve!(solver)
                println("STARTING CLARABEL BNB LOOP ")
                root, term_status = branch_and_bound_solve(solver, result,n,ϵ)
                @test term_status  == "INFEASIBLE"

            end
        end
    end
end