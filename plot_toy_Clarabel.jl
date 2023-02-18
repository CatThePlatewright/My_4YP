#Plot comparison
using JLD
using PyPlot
using Printf

k_list = [3]

ind = 2:20


PyPlot.clf()
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
fig = figure()
color_set = ["green", "orange", "black","cyan","red"]
marker_set = ["^" "s" "D" "x"]
handles = Vector()
for i in 1:lastindex(k_list)
    #= with_iter = load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"with_iter")
    without_iter= load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"without_iter")
    first_iter_num = load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"first_iter_num") 
    total_nodes = load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"total_nodes_num")
    total_nodes_without = load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"total_nodes_without_num") =#
    with_iter = load(@sprintf("my_toy_k=%d.jld",k_list[i]),"with_iter")
    without_iter= load(@sprintf("my_toy_k=%d.jld",k_list[i]),"without_iter")
    first_iter_num = load(@sprintf("my_toy_k=%d.jld",k_list[i]),"first_iter_num")
    total_nodes = load(@sprintf("my_toy_k=%d.jld",k_list[i]),"total_nodes")
    total_nodes_without = load(@sprintf("my_toy_k=%d.jld",k_list[i]),"total_nodes_without")
    with_iter = with_iter./total_nodes
    without_iter = without_iter./total_nodes_without
    percentage = (with_iter .- first_iter_num) ./(without_iter .- first_iter_num)
    total_percentage = sum(with_iter .- first_iter_num) / sum(without_iter .- first_iter_num)
    for j = 1:lastindex(percentage)
        if isnan(percentage[j])
            percentage[j] = 1.0
        end
    end
    subplot(211)
    #p1, = PyPlot.semilogy(ind , without_iter .- first_iter_num, linestyle="dashed",color= color_set[i],  marker = marker_set[i], markersize = 4, markevery = 1)
    #p2, = PyPlot.semilogy(ind, with_iter .- first_iter_num, color= color_set[i], label = @sprintf("k= %d",k_list[i]), marker = marker_set[i], markersize = 4, markevery = 1)
    p1, = PyPlot.step(ind , without_iter .- first_iter_num, linestyle="dashed",color= color_set[i],  marker = marker_set[i], markersize = 4, markevery = 1)
    p2, = PyPlot.step(ind, with_iter .- first_iter_num, color= color_set[i], label = @sprintf("k= %d",i), marker = marker_set[i], markersize = 4, markevery = 1)
    annotate("Dashed - No early termination",(4,5300))
    annotate("Solid - With early termination",(4,1000))
    ylabel("# IPM iterations")
    xlim([2,20])
    push!(handles,p2)
    subplot(212)
    
    PyPlot.step(ind, percentage, color= color_set[i], marker = marker_set[i], markersize = 4)
    ylabel("Ratio")
    xlabel("Number of integer variables")
    xlim([2,20])
end
PyPlot.legend(handles=handles)

savefig("toy_problem_3k.pdf")
printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)

# fn = plot(ind .- start_idx, [without_iter5 .- first_iter_num5, with_iter5 .- first_iter_num5], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)

