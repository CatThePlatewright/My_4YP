#= Defining the node and node-data making up the tree. 
Binary Tree at the beginning for first bnb example 
    =#

using AbstractTrees, Gurobi, JuMP

#The Node is for later implementation of tree nodes with multiple children (integer, not only binary variables)
#=struct Node{MyNodeData}
    data::MyNodeData
    children::Vector{Node{MyNodeData}}

    Node{MyNodeData}(data, ch) where {MyNodeData} = new{MyNodeData}(data, collect(Node{MyNodeData}, ch))
end
nodevalue(n::Node{MyNodeData}) = n.value

children(n::Node{MyNodeData}) = n.children =#

#= THIS IS FROM BnbNode EXAMPLE IN PKG AbstractTrees =#
mutable struct BnbNode{MyNodeData}
    data::MyNodeData #MyNodeData type
    parent::Union{Nothing,BnbNode{MyNodeData}}
    left::Union{Nothing,BnbNode{MyNodeData}}
    right::Union{Nothing,BnbNode{MyNodeData}}

    function BnbNode{MyNodeData}(data, parent=nothing, l=nothing, r=nothing) where MyNodeData
        new{MyNodeData}(data, parent, l, r)
    end
end
BnbNode(data) = BnbNode{typeof(data)}(data)

function leftchild!(parent::BnbNode, data)
    isnothing(parent.left) || error("left child is already assigned")
    node = typeof(parent)(data, parent)
    parent.left = node
end
function rightchild!(parent::BnbNode, data)
    isnothing(parent.right) || error("right child is already assigned")
    node = typeof(parent)(data, parent)
    parent.right = node
end

## Things we need to define
function AbstractTrees.children(node::BnbNode)
    if isnothing(node.left) && isnothing(node.right)
        ()
    elseif isnothing(node.left) && !isnothing(node.right)
        (node.right,)
    elseif !isnothing(node.left) && isnothing(node.right)
        (node.left,)
    else
        (node.left, node.right)
    end
end

AbstractTrees.nodevalue(n::BnbNode) = n.data

AbstractTrees.ParentLinks(::Type{<:BnbNode}) = StoredParents()

AbstractTrees.parent(n::BnbNode) = n.parent

AbstractTrees.NodeType(::Type{<:BnbNode{MyNodeData}}) where {MyNodeData} = HasNodeType()
AbstractTrees.nodetype(::Type{<:BnbNode{MyNodeData}}) where {MyNodeData} = BnbNode{MyNodeData}

"
model is the JuMP model with underlying solver, points to the same model, so model[:x] is always the same
solution_x: you need the solution stored in separate field as model[:x] holds only the solution to the current model
fixed_x_ind:  which x's are fixed on this node, stored in vector of length 0 (root) - height_of_tree (leaves)
fixed_x_values: which (binary) value the corresponding x's are fixed to
lb, ub are the lower and upper bounds on the objective computed for this node"
mutable struct MyNodeData #mutable since lb and ub can be updated after first creation of node
     model #the JuMP model 
     depth::Int
     solution_x::Vector{Float64} # storing the BEST solution (resulting in best lowest ub)
     fixed_x_ind::Vector{Int} 
     fixed_x_values::Vector{Float64} # to which value is it bounded at branching
     bounds::Vector{String} # storing whether variable[fixed_x_ind] is bounded by "ub" or "lb" to fixed_x_value
     lb::Float64
     ub::Float64
     function MyNodeData(model, solution_x, fixed_x_ind,fixed_x_values,bounds, lb,ub) 
        return new(model, length(fixed_x_ind), solution_x, fixed_x_ind,fixed_x_values,bounds, lb,ub)
     end
 end


""" Data fields for a BnBNode within the Clarabel implementation"""
 mutable struct ClarabelNodeData #mutable since lb and ub can be updated after first creation of node
    solver #the Clarabel solver object in Clarabel 
    is_pruned::Bool 
    solution_x::Vector{Float64} # solution x to relaxed problem of node
    solution_z::Vector{Float64} # solution z to relaxed problem of node
    solution_s::Vector{Float64} # solution s to relaxed problem of node
    depth::Int
    fixed_x_ind::Vector{Int} 
    fixed_x_values::Vector{Float64} # to which value is it bounded 
    upper_or_lower_vec::Vector{Int}
    lb::Float64 # on objective_value
    function ClarabelNodeData(solver, solution_x,solution_z, solution_s,fixed_x_ind,fixed_x_values, upper_or_lower_vec, lb) 
       return new(solver, false, deepcopy(solution_x), deepcopy(solution_z),deepcopy(solution_s), length(fixed_x_ind),fixed_x_ind,fixed_x_values, upper_or_lower_vec, lb)
    end
end

 function update_best_lb(node::BnbNode) #just update root not all parents???
    while ~isroot(node)
        AbstractTrees.parent(node).data.lb = node.data.lb
        node = AbstractTrees.parent(node)
    end
end

function branch_from_node(node::BnbNode)
    if isnothing(node.left) || isnothing(node.right) 
        return node
    elseif node.left.data.is_pruned && node.right.data.is_pruned
        println("Both children pruned, so prune this node")
        node.data.is_pruned = true
        branch_from_node(node.parent)
    elseif node.left.data.is_pruned && ~node.right.data.is_pruned
        branch_from_node(node.right)
    elseif node.right.data.is_pruned && ~node.left.data.is_pruned
        branch_from_node(node.left)
    else
        if (node.left.data.lb <= node.right.data.lb) 
            println("left child lb: ", node.left.data.lb, " right child lb: ",node.right.data.lb)
            node = node.left
            println("BRANCHING LEFT AT DEPTH ", node.data.depth)
        elseif  (node.left.data.lb > node.right.data.lb) 
            println("left child lb: ", node.left.data.lb, " right child lb: ",node.right.data.lb)
            node = node.right
            println("BRANCHING RIGHT AT DEPTH ", node.data.depth)
        end
        branch_from_node(node)

    end
end 



