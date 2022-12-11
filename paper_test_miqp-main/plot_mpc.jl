#Plot comparison
using JLD
using PyPlot

with_iter = load("data\\mimpc_comp_N=2.jld", "with_iter")
without_iter = load("data\\mimpc_comp_N=2.jld", "without_iter")
first_iter_num = load("data\\mimpc_comp_N=2.jld", "first_iter_num")

# start_idx = 1
# end_idx = 2400
start_idx = 2200
end_idx = 2300
ind = start_idx:end_idx

percentage = (with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]) ./(without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx])
total_percentage = sum(with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]) / sum(without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx])
for i = 1:length(percentage)
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
ylabel("# QP iterations")
xlim([0,100])
PyPlot.legend(handles = [p1, p2])
subplot(212)
PyPlot.step(ind .- start_idx, ones(end_idx - start_idx + 1), color= "black", marker = "o", markersize = 4)
PyPlot.step(ind .- start_idx, percentage, color= "red", marker = "x", markersize = 4)
ylabel("Ratio")
xlabel("Intervals")
xlim([0,100])
savefig("data\\mpc_comparison-N=2.pdf")

# fn = plot(ind .- start_idx, [without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
#
# # savefig(fn, "mpc_comparison")
# # savefig(fn_per, "mpc_comparison-percentage")
#
# l = @layout [a ; b]
# fn_comb = plot(fn, fn_per, fmt = :eps, layout = l)
# savefig(fn_comb, "mpc_comparison")