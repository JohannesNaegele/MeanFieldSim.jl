using MeanFieldSim
using StatsPlots
using CUDA
using Flux
CUDA.functional()

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
# FIXME:
hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook(), ValidationHook(20)])
@profview approximate(game; n=2, N=50000, p=2, iterations=1, hook)
# hook[4].training_error
# hook[4].test_error

p = density(title="Stochastic discount factor");
for i in eachindex(results)
    density!(results[i].ξ, label="Iteration $i")
end
p

p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(results)
    density!(results[i].ψ, label="Iteration $i")
end
p

time = collect(eachindex(results[1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(results)
    p = plot!(time, results[i].ψ_repr, label="Iteration $i");
end
p

# different λ
results = []
for λ in [0.0, 0.2, 0.4]
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
    hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook(), ValidationHook(20)])
    approximate(game; n=20, N=50000, p=2, iterations=4, hook)
    push!(results, hook[3].results)
end

# different γ
for γ in [0.15, 0.3, 0.45]
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
    hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook(), ValidationHook(20)])
    approximate(game; n=20, N=50000, p=2, iterations=4, hook)
    push!(results, hook[3].results)
end

for ρ in [0.0, 0.25, 0.5, 0.75]
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
    hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook(), ValidationHook(20)])
    approximate(game; n=20, N=50000, p=2, iterations=4, hook)
    push!(results, hook[3].results)
end

# Plots for λ
p = density(title="Total average emissions", xlims=[0,5]);
for i in eachindex(results)[1:3]
    density!(results[i][3].results.ψ, label="Iteration $i")
end
p

time = collect(eachindex(results[1].ψ_repr))
p = plot(title="Expected emissions as function of time");
for i in eachindex(results)[1:3]
    p = plot!(time, results[i].ψ_repr, label="Iteration $i");
end
p

# # Syntax, die ich gerne hätte
# for n in 1:20
#     game.dings = 1.0/n
#     approximate(game; n=20, N=50000, p=2, iterations=10, hook)
# end