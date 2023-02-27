#Plot comparison
using JLD
using PyPlot

with_iter = load("mimpc_iterations_N=8_λ=0.99.jld","with_iter")
without_iter = load("mimpc_iterations_N=8_λ=0.99.jld", "without_iter")
first_iter_num = load("mimpc_iterations_N=8_λ=0.99.jld", "first_iter_num")
with_iter2 = load("mimpc_iterations_N=8.jld","with_iter")
without_iter2 = load("mimpc_iterations_N=8.jld", "without_iter")
first_iter_num2 = load("mimpc_iterations_N=8.jld", "first_iter_num")

start_idx = 1
end_idx = 100
ind = start_idx:end_idx

percentage = (with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]) ./(without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx])
total_percentage = sum(with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]) / sum(without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx])
for i = 1:lastindex(percentage)
    if isnan(percentage[i])
        percentage[i] = 1.0
    end
end
percentage2 = (with_iter2[start_idx:end_idx] .- first_iter_num2[start_idx:end_idx]) ./(without_iter2[start_idx:end_idx] .- first_iter_num2[start_idx:end_idx])
total_percentage2 = sum(with_iter2[start_idx:end_idx] .- first_iter_num2[start_idx:end_idx]) / sum(without_iter2[start_idx:end_idx] .- first_iter_num2[start_idx:end_idx])
for i = 1:lastindex(percentage2)
    if isnan(percentage2[i])
        percentage2[i] = 1.0
    end
end


PyPlot.clf()
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
fig = figure()
subplot(211)
# color_set = ["green" "orange" "black" "cyan"]
# marker_set = ["^" "s" "D" "x"]
p1, = PyPlot.step(ind .- start_idx, without_iter2[start_idx:end_idx] .- first_iter_num2[start_idx:end_idx], color= "black", label = "No early termination, cold start", marker = "o", markersize = 4, markevery = 1)
p2, = PyPlot.step(ind .- start_idx, with_iter2[start_idx:end_idx] .- first_iter_num2[start_idx:end_idx], color= "red", label = "With early termination, cold start", marker = "x", markersize = 4, markevery = 1)
p3, = PyPlot.step(ind .- start_idx, without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], color= "grey", label = "No early termination, warm start", marker = "s", markersize = 4, markevery = 1)
p4, = PyPlot.step(ind .- start_idx, with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], color= "green", label = "With early termination, warm start", marker = "^", markersize = 4, markevery = 1)

ylabel("# IPM iterations")
xlim([0,100])
PyPlot.legend(handles = [p1, p2, p3,p4])
subplot(212)
PyPlot.step(ind .- start_idx, ones(end_idx - start_idx + 1), color= "black", marker = "o", markersize = 4)
PyPlot.step(ind .- start_idx, percentage2, color= "red",marker = "x", markersize = 4)
PyPlot.step(ind .- start_idx, percentage, color= "green", marker = "^", markersize = 4)

ylabel("Ratio")
xlabel("Intervals")
xlim([0,100])

savefig("mpc_comparison_N=8_warmvscold.pdf")

# fn = plot(ind .- start_idx, [without_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx], with_iter[start_idx:end_idx] .- first_iter_num[start_idx:end_idx]], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
#
# # savefig(fn, "mpc_comparison")
# # savefig(fn_per, "mpc_comparison-percentage")
#
# l = @layout [a ; b]
# fn_comb = plot(fn, fn_per, fmt = :eps, layout = l)
# savefig(fn_comb, "mpc_comparison")
