cd("./MeanFieldSim")
using Revise
import Pkg; Pkg.activate("./")
using MeanFieldSim
using StatsPlots
using Flux

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
approximate(game; n=20, N=50000, p=2, iterations=4, hook)
results = hook[3].results
# hook[4].training_error
# hook[4].test_error
p = density(results[1].ξ);
for i in eachindex(results)[2:end]
    density!(results[i].ξ)
end
p

p = density(results[1].ψ, xlims=[0,5]);
for i in eachindex(results)[2:end]
    density!(results[i].ψ)
end
p

time = collect(eachindex(results[1].ψ_repr))
p = plot(time, results[1].ψ_repr);
for i in eachindex(results)[2:end]
    p = plot!(time, results[i].ψ_repr);
end
p

# # Syntax, die ich gerne hätte
# for n in 1:20
#     game.dings = 1.0/n
#     approximate(game; n=20, N=50000, p=2, iterations=10, hook)
# end