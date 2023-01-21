#Plot comparison
using JLD
using PyPlot

with_iter = load("toy_problem.jld","with_iter")
without_iter = load("toy_problem.jld", "without_iter")
first_iter_num = load("toy_problem.jld", "first_iter_num")

start_idx = 1
end_idx = lastindex(with_iter)
ind = start_idx:end_idx

percentage = (with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]) ./(without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx])
total_percentage = sum(with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]) / sum(without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx])
for i = 1:lastindex(percentage)
    if isnan(percentage[i])
        percentage[i] = 1.0
    end
end

PyPlot.clf()
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
fig = figure()
subplot(211)
# color_set = ["green" "orange" "black" "cyan"]
# marker_set = ["^" "s" "D" "x"]
p1, = PyPlot.step(ind .- start_idx, without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], color= "black", label = "No early termination", marker = "o", markersize = 4, markevery = 1)
p2, = PyPlot.step(ind .- start_idx, with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], color= "red", label = "With early termination", marker = "x", markersize = 4, markevery = 1)
ylabel("# IP iterations")
xlim([0,length(with_iter)])
PyPlot.legend(handles = [p1, p2])
subplot(212)
PyPlot.step(ind .- start_idx, ones(end_idx - start_idx + 1), color= "black", marker = "o", markersize = 4)
PyPlot.step(ind .- start_idx, percentage, color= "red", marker = "x", markersize = 4)
ylabel("Ratio")
xlabel("Intervals")
xlim([0,length(with_iter)])
savefig("toy_problem.pdf")
printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)

# fn = plot(ind .- start_idx, [without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
#
# # savefig(fn, "mpc_comparison")
# # savefig(fn_per, "mpc_comparison-percentage")
#
# l = @layout [a ; b]
# fn_comb = plot(fn, fn_per, fmt = :eps, layout = l)
# savefig(fn_comb, "mpc_comparison")
