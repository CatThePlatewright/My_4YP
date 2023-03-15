using JLD
using PyPlot
using Printf


ρ_values_str = ["1e-7", "1e-3",  "1e-1", "1","10","100"]
ρ_values_str = ["1"]

T = 2000
PyPlot.clf()
fig = figure("Portfolio value over time",figsize=(10,10))

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
color_set = ["green", "orange", "black","cyan","red","blue"]

handles = Vector()
xaxis_range = 1:T

for i in 1:lastindex(ρ_values_str)
    portfolio_value = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"Vt")
    xplus = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"xplus")
    xminus = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"xminus")
    r_solution = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"r_solution")
    println("xplus: ", round.(xplus,digits=2))
    println("xminus: ", round.(xminus,digits=2))
    println("r: ", round.(r_solution,digits=2))

    p, = plot(xaxis_range, portfolio_value/1000, color = color_set[i],label=@sprintf("ρ = %s",ρ_values_str[i]))
    push!(handles,p)
end
PyPlot.legend(handles=handles)
ylabel("Portfolio value (in thousand dollars)")
xlabel("Day") 
ylim([0,150])
grid()

savefig("my_total_portfolio_.pdf")

# Complete plot
#= ax[0].set_ylabel()
ax[0].set_xlabel("Day")
ax[0].set_yticks([10, 50, 100, 150])
ax[0].set_xlim([0, T])
ax[0].grid()

 =#



    
