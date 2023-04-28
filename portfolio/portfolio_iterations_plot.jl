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

with_iter = load("portfolio_iterations_1.jld","with_iter")
without_iter = load("portfolio_iterations_1.jld","without_iter")
first_iter_num = load("portfolio_iterations_1.jld","first_iter_num") 
total_nodes = load("portfolio_iterations_1.jld","total_nodes")
total_nodes_without = load("portfolio_iterations_1.jld","total_nodes_without")
println(length(total_nodes))
ind = 1000:(1000+lastindex(with_iter)-1)
with_iter =(with_iter .- first_iter_num)
without_iter = (without_iter .- first_iter_num)
percentage = (with_iter)./(without_iter)
total_percentage = sum(with_iter) / sum(without_iter)
for j = 1:lastindex(percentage)
    if isnan(percentage[j])
        percentage[j] = 1.0
    end
end
subplot(211)
p1, = PyPlot.plot(ind , without_iter, color= "black", label="No early termination", markersize = 4, markevery = 1)
p2, = PyPlot.plot(ind, with_iter, color= "red", label="With early termination", markersize = 4, markevery = 1)
#= annotate("Dashed - No early termination",(2.4,6))
annotate("Solid - With early termination",(2.4,5)) =#
ylabel("# IPM iterations")
PyPlot.legend(handles=[p1,p2])
xlim([1000,1000+ lastindex(with_iter)-1])
subplot(212)

PyPlot.plot(ind, ones(length(ind)), color= "black", markersize = 4)
PyPlot.plot(ind, percentage, color= "red", markersize = 4)
ylabel("Ratio")
xlabel("Days")
xlim([1000,1000+ lastindex(with_iter)-1])

savefig("portfolio_early_termination_rho=1.pdf")
printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)

# fn = plot(ind .- start_idx, [without_iter5 .- first_iter_num5, with_iter5 .- first_iter_num5], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)

