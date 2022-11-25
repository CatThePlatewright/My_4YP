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

#= THIS IS FROM BINARYNODE EXAMPLE IN PKG AbstractTrees =#
mutable struct BinaryNode{MyNodeData}
    data::MyNodeData #MyNodeData type
    parent::Union{Nothing,BinaryNode{MyNodeData}}
    left::Union{Nothing,BinaryNode{MyNodeData}}
    right::Union{Nothing,BinaryNode{MyNodeData}}

    function BinaryNode{MyNodeData}(data, parent=nothing, l=nothing, r=nothing) where MyNodeData
        new{MyNodeData}(data, parent, l, r)
    end
end
BinaryNode(data) = BinaryNode{typeof(data)}(data)

function leftchild!(parent::BinaryNode, data)
    isnothing(parent.left) || error("left child is already assigned")
    node = typeof(parent)(data, parent)
    parent.left = node
end
function rightchild!(parent::BinaryNode, data)
    isnothing(parent.right) || error("right child is already assigned")
    node = typeof(parent)(data, parent)
    parent.right = node
end

## Things we need to define
function AbstractTrees.children(node::BinaryNode)
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

AbstractTrees.nodevalue(n::BinaryNode) = n.data

AbstractTrees.ParentLinks(::Type{<:BinaryNode}) = StoredParents()

AbstractTrees.parent(n::BinaryNode) = n.parent

AbstractTrees.NodeType(::Type{<:BinaryNode{MyNodeData}}) where {MyNodeData} = HasNodeType()
AbstractTrees.nodetype(::Type{<:BinaryNode{MyNodeData}}) where {MyNodeData} = BinaryNode{MyNodeData}

"
model is the JuMP model with underlying solver, points to the same model, so model[:x] is always the same
solution_x: you need the solution stored in separate field as model[:x] holds only the solution to the current model
fixed_x_ind:  which x's are fixed on this node, stored in vector of length 0 (root) - height_of_tree (leaves)
fixed_x_values: which (binary) value the corresponding x's are fixed to
lb, ub are the lower and upper bounds computed for this node"
mutable struct MyNodeData #mutable since lb and ub can be updated after first creation of node
     model #the JuMP model 
     depth::Int
     solution_x::Vector{Float64} # storing the BEST solution (best ub)
     fixed_x_ind::Vector{Int} 
     fixed_x_values::Vector{Float64} # to which value is it fixed (0 or 1)
     lb::Float64
     ub::Float64
     function MyNodeData(model, solution_x, fixed_x_ind,fixed_x_values,lb,ub) 
        return new(model, length(fixed_x_ind), solution_x, fixed_x_ind,fixed_x_values,lb,ub)
     end
 end

 mutable struct ClarabelNodeData #mutable since lb and ub can be updated after first creation of node
    solver #the Clarabel solver object in Clarabel 
    solution # the result/solution object != solution_x
    depth::Int
    solution_x::Vector{Float64} # storing the BEST solution (best ub)
    fixed_x_ind::Vector{Int} 
    fixed_x_values::Vector{Float64} # to which value is it fixed (0 or 1)
    lb::Float64
    ub::Float64
    debug_b::Vector{Float64} # stores only the b-vector in compute_ub (so includes rounded relaxed_vars)
    function ClarabelNodeData(solver, solution, solution_x, fixed_x_ind,fixed_x_values,lb,ub) 
       return new(solver, solution, length(fixed_x_ind), solution_x, fixed_x_ind,fixed_x_values,lb,ub, zeros(1))
    end
end

 function update_best_lb(node::BinaryNode) #just update root not all parents???
    while ~isroot(node)
        AbstractTrees.parent(node).data.lb = node.data.lb
        node = AbstractTrees.parent(node)
    end
end

function update_best_ub(node::BinaryNode)
    root = getroot(node)
    if ~isroot(node) && (node.data.ub < root.data.ub)
        root.data.ub = node.data.ub 
        println("FOUND BETTER UB AT DEPTH ", node.data.depth)
        root.data.solution_x = node.data.solution_x
    end
end

function branch_from_node(node::BinaryNode)
    if isnothing(node.left) || isnothing(node.right) 
        return node
    elseif (node.left.data.lb <= node.right.data.lb) 
        println("left child lb: ", node.left.data.lb, " right child lb: ",node.right.data.lb)
        node = node.left
        println("BRANCHING LEFT AT DEPTH ", node.data.depth)
    elseif  (node.left.data.lb > node.right.data.lb) 
        println("left child lb: ", node.left.data.lb, " right child lb: ",node.right.data.lb)
        node = node.right
        println("BRANCHING RIGHT AT DEPTH ", node.data.depth)
    else
        error("Error with Tree?")
        print_tree(node)
        return 
    end
    branch_from_node(node)
end 



