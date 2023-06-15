#Plot comparison
using JLD
using PyPlot, Printf, Statistics
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
color_set = ["red" "green" "orange" "black" "cyan"]
marker_set = ["^" "s" "D" "x"]

function plot_mpc(n::Int,sparsity)
    ind = load(@sprintf("mpc_%s_N=%d.jld",sparsity,n),"ind")
    total_time = load(@sprintf("mpc_%s_N=%d.jld",sparsity,n), "total_time")
    total_time_without = load(@sprintf("mpc_%s_N=%d.jld",sparsity,n), "total_time_without")
    percentage = load(@sprintf("mpc_%s_N=%d.jld",sparsity,n), "percentage_ratio")
    
    PyPlot.clf()
    fig = figure()


    fig1 = subplot(211)
    p1, = fig1.step(ind, total_time, color= "black", label = "No early termination", markersize = 4, markevery = 1)
    p2, = fig1.step(ind, total_time_without, color= "red", label = "With early termination", markersize = 4, markevery = 1)

    ylabel("Time (ms)")
    xlim([0,100])
    #ylim([0,150])

    fig1.legend(handles = [p1, p2],loc="upper left")

    fig2 = subplot(212)
    fig2.step(ind, ones(length(ind)), color= "black", )
    fig2.step(ind, percentage, color= "red")

    ylabel("Ratio")
    xlabel("Intervals")
    xlim([0,100])
    # ylim([0.3,1.1])
    fig.savefig(@sprintf("mpc_%s_N=%d.pdf",sparsity,n))
    
end