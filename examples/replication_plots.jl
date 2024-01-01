using JLD2
using StatsPlots

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