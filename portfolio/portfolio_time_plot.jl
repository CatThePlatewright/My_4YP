#Plot comparison
using JLD
using PyPlot
using Printf

PyPlot.clf()
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
fig = figure()
color_set = ["green", "orange", "black","cyan","red"]
marker_set = ["^" "s" "D" "x"]

percentage_without_time = load("portfolio_time_0.001.jld","percentage_without_time")
percentage_with_time = load("portfolio_time_0.001.jld","percentage_with_time")
total_time = load("portfolio_time_0.001.jld","total_time") 
total_time_without = load("portfolio_time_0.001.jld","total_time_without") 
ind = load("portfolio_time_0.001.jld","ind") 

PyPlot.clf()
fig = figure()

fig1 = subplot(211)
p1, = fig1.step(ind , total_time_without, color= "black", label="No early termination", markersize = 4, markevery = 1)
p2, = fig1.step(ind, total_time, color= "red", label="With early termination", markersize = 4, markevery = 1)
#= annotate("Dashed - No early termination",(2.4,6))
annotate("Solid - With early termination",(2.4,5)) =#
ylabel("Time (ms)")
PyPlot.legend(handles=[p1,p2])
xlim([0,length(total_time)-1])
# yscale("log")
fig2 = subplot(212)

fig2.step(ind, percentage_without_time, color= "black", markersize = 4)
fig2.step(ind, percentage_with_time, color= "red", markersize = 4)
ylabel("Ratio")
xlabel("Days")
xlim([0,length(total_time)-1])

fig.savefig("portfolio_early_termination_K=10.pdf")
printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)

# fn = plot(ind .- start_idx, [without_iter5 .- first_iter_num5, with_iter5 .- first_iter_num5], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)

