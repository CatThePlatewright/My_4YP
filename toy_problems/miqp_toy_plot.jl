#Plot comparison
using JLD
using PyPlot
using Printf



k_list = [0.50,0.25]
ind = 2:25

PyPlot.clf()
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
fig = figure()
color_set = ["green", "orange", "black","cyan","red"]
marker_set = ["^" "s" "D" "x"]
handles = Vector()

for i in 1:lastindex(k_list)
    with_iter = load(@sprintf("MIQP_toy_k=%1.2f.jld",k_list[i]),"with_iter")
    without_iter= load(@sprintf("MIQP_toy_k=%1.2f.jld",k_list[i]),"without_iter")
    first_iter_num = load(@sprintf("MIQP_toy_k=%1.2f.jld",k_list[i]),"first_iter_num") 
    total_nodes = load(@sprintf("MIQP_toy_k=%1.2f.jld",k_list[i]),"total_nodes")
    total_nodes_without = load(@sprintf("MIQP_toy_k=%1.2f.jld",k_list[i]),"total_nodes_without")
    replace!(total_nodes,0=>1)
    replace!(total_nodes_without,0=>1)

    println(total_nodes,total_nodes_without)
    with_iter =(with_iter .- first_iter_num)./total_nodes
    without_iter = (without_iter .- first_iter_num)./total_nodes_without
    percentage = (with_iter)./(without_iter)
    total_percentage = sum(with_iter) / sum(without_iter)
    for j = 1:lastindex(percentage)
        if isnan(percentage[j])
            percentage[j] = 1.0
        end
    end
    subplot(211)
    p1, = PyPlot.step(ind , without_iter, linestyle="dashed",color= color_set[i],  marker = marker_set[i], markersize = 4, markevery = 1)
    p2, = PyPlot.step(ind, with_iter, color= color_set[i], label=@sprintf("k= floor(%1.2f*n)",k_list[i]), marker = marker_set[i], markersize = 4, markevery = 1)
    annotate("Dashed - No early termination",(17,3))
    annotate("Solid - With early termination",(17,2))
    ylabel("# IPM iterations (average per node)")
    xlim([2,25])
    push!(handles,p2)
    subplot(212)

    PyPlot.step(ind, percentage, color= color_set[i], marker = marker_set[i], markersize = 4)
    ylabel("Ratio")
    xlabel("Number of x variables (n)")
    xlim([2,25])
end
PyPlot.legend(handles=handles)

savefig("MIQP_plot.pdf")
printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)

# fn = plot(ind .- start_idx, [without_iter5 .- first_iter_num5, with_iter5 .- first_iter_num5], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)

