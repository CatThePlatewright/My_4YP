using JuMP, Gurobi, LinearAlgebra, Random, DataStructures
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "OutputFlag", 0)


# model parameters
n = 4
k = 2
Q = Matrix{Float16}(I, n, n) 
Random.seed!(1234)
c = rand(Float16,n)

mutable struct node{T}
    data::T #tuple of index of variable, binary value, then the objective value from solution
    left::Union{node, Nothing}
    right::Union{node, Nothing}
    node(T)= (data=T; left= Nothing; right= Nothing) 
end
pq = PriorityQueue{Int,node}()
function add_constraints(
    model::Model, k::Int, lb, ub)
    x = model[:x]
    @constraint(model, sum_constraint, sum(x) == k)
    @constraint(model, lb_constraint, x .>= lb)
    @constraint(model, ub_constraint, x .<= ub)

    return
end

function fix_variable(model::Model, i::Int, value::Float16)
    x = model[:x]
    con1 = model[:lb_constraint]
    con2 = model[:ub_constraint]
    set_normalized_rhs(con1[i], value)
    set_normalized_rhs(con2[i], value)
end

function add_variables(
    model::Model,n::Int, binary)
    if binary
        return @variable(model, x[1:n], Bin)
    end
    return @variable(model, x[1:n]) #free variables, then add constraint vector
end

function build_base_model(model::Model, n::Int,k::Int,Q::Matrix,c::Vector, binary = false)
    x = add_variables(model, n, binary)
    @objective(model, Min, x'*Q*x + c'*x)
    add_constraints(model, k, zeros(n), ones(n)) # binary case would make ub and lb constraints redundant!
    return model
end

function build_child_model(model::Model, i::Int, fix_to_value, final_p_values)
    set_optimizer(model, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
    fix_variable(model, i, Float16(fix_to_value))
        # Or: new_model, reference_map = copy_model(model)
        #  x_new = reference_map[x]
    x = model[:x]
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        if i == n
            println(model)
            println("Optimal values for x:", value.(x))
            println("Optimal Objective", objective_value(model))
            push!(final_p_values, objective_value(model))
        end
        return objective_value(model)
    else
        return Inf
    end
    
end

function solve_model(parent_model::Model, i::Int, p_values::Vector{Float64}, final_p_values::Vector{Float64})
    left_model = copy(parent_model)
    right_model = copy(parent_model)
    push!(p_values, build_child_model(left_model, i, 0.0,final_p_values))
    push!(p_values, build_child_model(right_model, i, 1.0, final_p_values))

    ### try recursive structure ###
    if i < n
        solve_model(left_model,i+1,p_values, final_p_values)
        solve_model(right_model,i+1,p_values, final_p_values)
    end
    return
end
p = Vector{Float64}() # DON'T FORGET () TO CREATE AN INSTANCE NOT JUST DATA TYPE!
final_p_values = Vector{Float64}()
#constraint_list =  Vector{Any}()

base_model = build_base_model(model,n,k,Q,c)
x = base_model[:x]
optimize!(base_model)
push!(p,objective_value(base_model))

solve_model(base_model,1,p, final_p_values)
