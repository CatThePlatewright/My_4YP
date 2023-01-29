#Plot comparison
using JLD
using PyPlot
using Printf

k_list = [-1,5]

ind = 3:20


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
    with_iter = load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"with_iter")
    without_iter= load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"without_iter")
    first_iter_num = load(@sprintf("my_toy_k=%d_warmstart.jld",k_list[i]),"first_iter_num")
    percentage = (with_iter .- first_iter_num) ./(without_iter .- first_iter_num)
    total_percentage = sum(with_iter .- first_iter_num) / sum(without_iter .- first_iter_num)
    for j = 1:lastindex(percentage)
        if isnan(percentage[j])
            percentage[j] = 1.0
        end
    end
    subplot(211)
    #= if i==1
        p1, = PyPlot.step(ind , without_iter[i] .- first_iter_num[i], color= "black", label = "No early termination", marker = "o", markersize = 4, markevery = 1)
        push!(handles,p1) 
    end =#
    p2, = PyPlot.semilogy(ind, with_iter .- first_iter_num, color= color_set[i], label = @sprintf("k= %d",k_list[i]), marker = marker_set[i], markersize = 4, markevery = 1)
    #p2, = PyPlot.step(ind, with_iter .- first_iter_num, color= color_set[i], label = @sprintf("k= %d",i), marker = marker_set[i], markersize = 4, markevery = 1)
    
    ylabel("# IPM iterations, semilog-scale")
    xlim([3,20])
    push!(handles,p2)
    subplot(212)
    if i == 1
        PyPlot.step(ind, ones(end_idx - start_idx + 1), color= "black", marker = "o", markersize = 4)
    end
    PyPlot.step(ind, percentage, color= color_set[i], marker = marker_set[i], markersize = 4)
    ylabel("Ratio")
    xlabel("Number of integer variables")
    xlim([3,20])
end
PyPlot.legend(handles=handles)

savefig("toy_problem_k=5_and_-1_logscale_warmstart.pdf")
printstyled("COPY AND SAVE DATA AND IMAGES UNDER DIFFERENT NAMES\n",color = :red)

# fn = plot(ind .- start_idx, [without_iter5 .- first_iter_num5, with_iter5 .- first_iter_num5], label = ["No early termination" "With early termination"], ylabel = "# QP iterations", marker = [:c :d], markershape = :auto, markersize = 2, seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)
# fn_per = plot(ind .- start_idx, [ones(end_idx - start_idx + 1), percentage], ylabel = "Ratio", marker = [:c :d], markersize = 2, label = "", seriestype=:step, linewidth = 1, color = [:black :orange], fmt = :eps)

