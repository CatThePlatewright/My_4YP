using SparseArrays, LinearAlgebra
using NPZ
using JLD

adaptive_data = npzread("results/adaptive_denseMPC_N=2.npz")

fixed_data = npzread("results/fixed_denseMPC_N=2.npz")