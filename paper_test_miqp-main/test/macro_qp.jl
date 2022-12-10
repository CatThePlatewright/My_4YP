# this shows how to load the maros meszaros problem data from the .jld2 file
using Revise
using FileIO, JLD2 # to load the matrices from the file
using LinearAlgebra, SparseArrays
includet("../src/admm_operator.jl")

# adjust accordingly
file_path = "D:/code/dataset/Archive/DUAL1.jld2"
data = load(file_path)

# the problem data is stored in the following format
# min 	0.5 x'Px + q'x + r
# s.t.  Af x == bf
#       l <= Ab x <= u

function new()
    Ab = data["Ab"]
    Af = data["Af"]
    bf = data["bf"]

    P = data["P"]
    q = data["q"]
    r = data["r"]

    l = data["l"]
    u = data["u"]

    A = vcat(Ab, Af)

    obj_true = data["obj_true"]
end
