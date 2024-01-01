using MeanFieldSim
using StatsPlots
using Flux
using JLD2

game = MeanFieldGame(
    γ_star=0.5,
    σ=0.1,
    γ=0.3,
    μ=0.05,
    V₀=1.0,
    C=0.7,
    C₂=1.0,
    λ=0.0,
    ρ=0.0,
    T=5.0
)

# TODO: adapt push! aka setindex
(h::ResultsHook)(::PostIterationStage, game, vars; kwargs...) = push!(
    h.results,
    (ξ=deepcopy(vars.ξ), ψ=deepcopy(vars.ψ), ψ_repr=deepcopy(vars.ψ_repr))
)

hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook(), ValidationHook(20)])
approximate(game; n=20, N=50000, p=2, iterations=4, hook)
first_results = hook[3].results

p = density(title="Stochastic discount factor");
for i in eachindex(first_results)
    density!(first_results[i].ξ, label="Iteration $i")
end
p

p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(first_results)
    density!(first_results[i].ψ, label="Iteration $i")
end
p

time = collect(eachindex(first_results[1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(first_results)
    p = plot!(time, first_results[i].ψ_repr, label="Iteration $i");
end
p

results = []

# different λ
hook = ResultsHook()
lambdas = [0.0, 0.2, 0.4]
@time for λ in lambdas
    game = MeanFieldGame(
        γ_star=0.5,
        σ=0.1,
        γ=0.3,
        μ=0.05,
        V₀=1.0,
        C=0.7,
        C₂=1.0,
        λ=λ,
        ρ=0.5,
        T=5.0
    )
    approximate(game; n=20, N=5000, p=2, iterations=2, hook)
end
push!(results, hook.results)

# different γ
hook = ResultsHook()
gammas = [0.15, 0.3, 0.45]
@time for γ in gammas
    game = MeanFieldGame(
        γ_star=0.5,
        σ=0.1,
        γ=γ,
        μ=0.05,
        V₀=1.0,
        C=0.7,
        C₂=1.0,
        λ=0.0,
        ρ=0.0,
        T=5.0
    )
    approximate(game; n=20, N=5000, p=2, iterations=2, hook)
end
push!(results, hook.results)

hook = ResultsHook()
rhos = [0.0, 0.25, 0.5, 0.75]
@time for ρ in rhos
    game = MeanFieldGame(
        γ_star=0.5,
        σ=0.1,
        γ=0.3,
        μ=0.05,
        V₀=1.0,
        C=0.7,
        C₂=1.0,
        λ=0.4,
        ρ=ρ,
        T=5.0
    )
    approximate(game; n=20, N=5000, p=2, iterations=2, hook)
end
push!(results, hook.results)

# Save results
jldsave("results.jld2"; first_results, results)

# Load results
data = load("results.jld2")
first_results = data["first_results"]
results = data["results"]

# Plots for γ

# Plots for λ
p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(results[1])
    density!(results[1][i].ψ, label="Iteration $i")
end
p

time = collect(eachindex(results[1][1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(results[1])
    p = plot!(time, results[1][i].ψ_repr, label="Iteration $i");
end
p

# Plots for ρ