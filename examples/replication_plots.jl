using JLD2
using StatsPlots
using LaTeXStrings
# using 

# Load results
data = load("results.jld2")
first_experiment = data["first_experiment"]
second_experiment = data["second_experiment"]
lambdas = data["lambdas"]
gammas = data["gammas"]
rhos = data["rhos"]

p = density(title="Stochastic discount factor");
for i in eachindex(first_experiment)
    density!(first_experiment[i].ξ, label="Iteration $i")
end
p

p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(first_experiment)
    density!(first_experiment[i].ψ, label="Iteration $i")
end
p

time = collect(eachindex(first_experiment[1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(first_experiment)
    p = plot!(time, first_experiment[i].ψ_repr, label="Iteration $i");
end
p

# Plots for λ
p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(second_experiment[1])
    density!(second_experiment[1][i].ψ, label=L"\lambda = %$(lambdas[i])")
end
p

time = collect(eachindex(second_experiment[1][1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(second_experiment[1])
    p = plot!(time, second_experiment[1][i].ψ_repr, label=L"\lambda = %$(lambdas[i])")
end
p

# Plots for γ
p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(second_experiment[2])
    density!(second_experiment[2][i].ψ, label=L"\gamma = %$(gammas[i])")
end
p

time = collect(eachindex(second_experiment[1][1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(second_experiment[2])
    p = plot!(time, second_experiment[2][i].ψ_repr, label=L"\gamma = %$(gammas[i])")
end
p

# Plots for ρ
p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(second_experiment[3])
    density!(second_experiment[3][i].ψ, label=L"\rho = %$(rhos[i])")
end
p

time = collect(eachindex(second_experiment[1][1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(second_experiment[3])
    p = plot!(time, second_experiment[3][i].ψ_repr, label=L"\rho = %$(rhos[i])")
end
p