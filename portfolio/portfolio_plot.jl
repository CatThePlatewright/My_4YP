using JLD
using PyPlot
using Printf


ρ_values_str = ["1e-4", "1e-3","1e-2","1.2e-2","1.5e-2","1e-1",]
# these give the same plot: 1/N,5e-2,4e-2,3e-2,1e-1,1
T = 2000
PyPlot.clf()
fig = figure("Portfolio value over time",figsize=(10,10))

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
color_set = ["green", "orange", "black","cyan","red","blue", "grey","pink","yellow"]

handles = Vector()
xaxis_range = 1:T

for i in 1:lastindex(ρ_values_str)
    portfolio_value = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"Vt")
    xplus = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"xplus")
    xminus = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"xminus")
    r_solution = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"r_solution")
    opt_value = load(@sprintf("portfolio_%s.jld",ρ_values_str[i]),"optimal_value")
    println("RHO: ",ρ_values_str[i])
    println("xplus: ", round.(xplus,digits=2))
    println("xminus: ", round.(xminus,digits=2))
    println("Optimal value: ",opt_value)

    p, = plot(xaxis_range, portfolio_value/1000, color = color_set[i],label=@sprintf("ρ = %s",ρ_values_str[i]))
    push!(handles,p)
end
PyPlot.legend(handles=handles)
ylabel("Portfolio value (in thousand dollars)")
xlabel("Day") 
grid()

savefig("my_total_portfolio.pdf")

# Complete plot
#= ax[0].set_ylabel()
ax[0].set_xlabel("Day")
ax[0].set_yticks([10, 50, 100, 150])
ax[0].set_xlim([0, T])
ax[0].grid()

 =#



    
