include("toy_bnb.jl")
using JLD, CSV

function create_variables(model,n,L)
    @variable(model,xplus[i = 1:n])
    @variable(model,xminus[i = 1:n])
    @variable(model,ζ[i=1:L])
    @variable(model,ρ)
    #return x, xplus, xminus, ζ, ρ
end
function add_constraints(model,n,L,Σ,smin)
    w = zeros(n) #initialise as 0 

    #lower bounds on variables
    @constraint(model,lxplus,-model[:xplus] .<=zeros(n))
    @constraint(model,lxminus,-model[:xminus] .<=zeros(n))
    @constraint(model,lzeta,-model[:ζ] .<=zeros(L))
    # upper bounds on variables
    @constraint(model,uxplus,model[:xplus] .<=ones(n))
    @constraint(model,uxminus,model[:xminus] .<=ones(n))
    @constraint(model,uzeta,model[:ζ] .<=ones(L))

    # smin is a threshold level: our total portfolio allocation in assets from sector k will be at least smin if ζ for that sector=1 (selected). This is to ensure we do not fall short.
    @constraint(model, cost_constraint, [ρ; (Σ^0.5)*(xplus+xminus)] in SecondOrderCone()) 
    @constraint(model, sum_constraint, sum(w+model[:xplus]-model[:xminus]) == 1.0)
    #=for z in ζ
        # some index set corresponding to z
        assets = 0
        for i in S
            assets += w[i]+ xplus[i]-xminus[i]
        end
        @constraint(model, sufficient_amount_con1, smin*z <= assets)
        @constraint(model, sufficient_amount_con2, smin+(1-smin)*ζ >= assets)
    end =#
    #Lmin = 1
    #@constraint(model, minimum_sectors, sum(ζ) >= Lmin)
    
end
function calc_variance(n)
    return Matrix(1.0I,n,n)
end

function get_return_data(n::Int)
    returns = CSV.File("portfolio_returns.csv")
    row1 = returns[1] # just get first row of returns at time t=0
    r̃ = zeros(n)
    companies = propertynames(row1)
    for col in 1:lastindex(companies)-1
        if col <= n
            r̃[col]= getproperty(row1,companies[col+1])
        end
    end
    return r̃
end

function solveGurobi()
    # check against Gurobi's mixed-integer solution
    exact_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(exact_model, "OutputFlag", 0)
    create_variables(exact_model,n,L) 
    set_binary.(exact_model[:ζ])
    @objective(exact_model, Min, 0.5*exact_model[:ρ]^2 + γ'*(exact_model[:xplus]-exact_model[:xminus]))
    add_constraints(exact_model,n,L,Λ,smin)
    println(exact_model)

    optimize!(exact_model)
    solution = vcat(value.(exact_model[:xplus]), value.(exact_model[:xminus]), value.(exact_model[:ζ]), value.(exact_model[:ρ]))
    println("Exact (Gurobi) solution: ", objective_value(exact_model) , " using ", solution) 

    return exact_model,solution
end
function getData()
    
    # create Clarabel instance, solve the fully relaxed base problem
    optimizer = Clarabel.Optimizer
    model = Model()
    setOptimizer(model, optimizer)
    create_variables(model,n,L)
    @objective(model, Min, 0.5*model[:ρ]^2 + γ'*(model[:xplus]-model[:xminus]))
    add_constraints(model,n,L,Λ,smin)
    optimize!(model)
    print("Solve_base_model in JuMP: ",solution_summary(model))
    println("JuMP solution relaxed: ",  objective_value(model) , " using ", value.(model[:xplus]), value.(model[:xminus]), value.(model[:ζ]), value.(model[:ρ]))
    P,q,A,b, cones= getClarabelData(model)
    println("P: ", P)
    println("q : ", q)
    println("A : ", A)
    println("b : ", b)
    println("cones : ", cones) 
    println("integer vars: ", binary_vars)
    return P,q,A,b,cones, binary_vars, exact_model
    
end

n = 2 # 20 is max. number of assets
L = 1 # about 6?
λ = 0.99
η = 100.0
ϵ = 1e-6
total_num = 2*n+L+1
Random.seed!(1234)
# γ = rand(Float64,n) 
γ = get_return_data(n)# return rates
Λ = calc_variance(n)
q = vcat(-γ,γ,zeros(L),0)
P = sparse([total_num],[total_num],[1.0],total_num,total_num)
smin = 0.1
binary_vars = collect(2*n+1:2*n+L) # indices of binary variables
without_iter_num = Int64[]
with_iter_num = Int64[]
first_iter_num = Int64[]
percentage_iter_reduction = Float64[]

P,q,A,b, s, binary_vars= getData()
exact_model, exact_solution = solveGurobi()
println("Setting up Clarabel solver...")
settings = Clarabel.Settings(verbose = false, equilibrate_enable = false, max_iter = 100)
solver   = Clarabel.Solver()

Clarabel.setup!(solver, P, q, A, b, s, settings)

result = Clarabel.solve!(solver, Inf)
println(result)

#start bnb loop
println("STARTING CLARABEL BNB LOOP ")

best_ub, feasible_solution, early_num, total_iter, fea_iter = branch_and_bound_solve(solver, result,n,ϵ, binary_vars,true,true,true,λ,η) 
println("Termination status of Clarabel solver:" , solver.info.status)
println("Found objective: ", best_ub, " using ", round.(feasible_solution,digits=3))
diff_sol_vector= round.(feasible_solution - value.(exact_solution),digits=5)
diff_solution=round(norm(diff_sol_vector),digits=5)
diff_obj = round(best_ub-objective_value(exact_model),digits=5)
if ~iszero(diff_solution) || ~iszero(diff_obj)
    println("Solution diff: ",diff_solution, "Obj difference: ", diff_obj)
    println(diff_sol_vector)
    println("index different value: ", [findall(x->x!=0,diff_sol_vector)])
    error("Solutions differ!")
end
println("Number of early terminated nodes: ", early_num)

# count QP iterations
printstyled("Total net iter num (with early_term_enable): ", total_iter-fea_iter, "\n", color = :green)
println(" ")  
#=     solver_without   = Clarabel.Solver()

Clarabel.setup!(solver_without, P, q, A, b,s, settings)

base_solution_without = Clarabel.solve!(solver_without)
best_ub_without, feasible_base_solution_without, early_num_without, total_iter_without, fea_iter_without = branch_and_bound_solve(solver_without, base_solution_without,n,ϵ, binary_vars, true, false, false,λ) 
println("Found objective without early_term: ", best_ub_without)
printstyled("Total net iter num (without): ", total_iter_without - fea_iter_without, "\n", color = :green)

println("Number of early terminated nodes (without): ", early_num_without)
reduction = 1 - (total_iter - fea_iter)/(total_iter_without - fea_iter_without)
println("Reduced iterations (percentage): ", reduction)
append!(without_iter_num, total_iter_without)
append!(with_iter_num, total_iter)
append!(percentage_iter_reduction, reduction)
if (fea_iter == fea_iter_without)
    append!(first_iter_num, fea_iter)
end   =#


printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)
#save("my_toy_k=10_warmstart.jld", "with_iter", with_iter_num, "without_iter", without_iter_num, "first_iter_num", first_iter_num, "percentage", percentage_iter_reduction)
