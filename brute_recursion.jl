#= This is my first attempt on solving a simple, cardinality constrained problem.
No upper bounds are calculated, i.e. the variables have not been rounded, due to an initial misunderstanding
of the methodology. However, since it is an exhaustive search approach, it still produces the wanted 
answer. Using recursion for depth-first traversal.  =#
using JuMP, Gurobi, LinearAlgebra, Random





function add_constraints(
    model::Model, k::Int)
    x = model[:x]
    @constraint(model, sum_constraint, sum(x) == k)
    return
end

function fix_variable(model::Model, i::Int, value::Float64)
    x = model[:x]
    @constraint(model, x[i]==value) # anonymous constraint to avoid constraintref/ name conflicts
    #@constraint(model, fix_var[i], x[i]==value)
    #push!(constraints,fix_var[i]) 
   # set_name(fix_var[i], "fix_var[$(length(constraints))]")
end

function add_variables(
    model::Model,n::Int, binary_vars)
    if ~isempty(binary_vars)
        x = @variable(model, 0.0 <= x[i = 1:n] <= 1.0)
        for bin in binary_vars
            set_binary(x[bin])
        end
#        model_x = model[:x]= @variable(model, [bin in binary_vars], Bin, base_name="binaries")
        #append!(model_x,@variable(model, [3], lower_bound=0.0, upper_bound=1.0,base_name="non_bin"))
        #append!(model_x, @variable(model, [i in setdiff(collect(1:n),binary_vars)], lower_bound=0.0, upper_bound= 1.0, base_name="non-binaries"))
        return x
    end
    return @variable(model, 0.0 <= x[i = 1:n] <= 1.0)
end

function build_base_model(model::Model, n::Int,k::Int,Q::Matrix,c::Vector, binary_vars = [])
    x = add_variables(model, n, binary_vars)
    @objective(model, Min, x'*Q*x + c'*x)
    add_constraints(model, k)
    return model
end

function build_child_model(model::Model, i::Int, fix_to_value, final_p_values)
    set_optimizer(model, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
    fix_variable(model, i, Float64(fix_to_value))
        # Or: new_model, reference_map = copy_model(model)
        #  x_new = reference_map[x]
    x = model[:x]
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        if i == n
            # println(model)
            #println("Optimal values for x:", value.(x))
            # println("Optimal Objective", objective_value(model))
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
    # ASK: should I empty!(parent_model) ???
    push!(p_values, build_child_model(left_model, i, 0.0,final_p_values))
    push!(p_values, build_child_model(right_model, i, 1.0, final_p_values))

    ### try recursive structure ###
    if i < n
        solve_model(left_model,i+1,p_values, final_p_values)
        solve_model(right_model,i+1,p_values, final_p_values)
    end
    return
end
#= p = Vector{Float64}() # DON'T FORGET () TO CREATE AN INSTANCE NOT JUST DATA TYPE!
final_p_values = Vector{Float64}()
#constraint_list =  Vector{Any}()

base_model = build_base_model(model,n,k,Q,c)
x = base_model[:x]
optimize!(base_model)
push!(p,objective_value(base_model)) =#

#solve_model(base_model,1,p, final_p_values)
