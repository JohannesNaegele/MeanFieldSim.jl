cd("./MeanFieldSim")
using Revise
import Pkg; Pkg.activate("./")
using MeanFieldSim
# using Plots
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

Base.@kwdef struct ResultsHook <: AbstractHook
    results=[]
end

# TODO: adapt push! aka setindex
(h::ResultsHook)(::PostIterationStage, game, vars; kwargs...) = push!(h.results, (ξ=vars.ξ,))

hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook()])
# hook[1]
results = hook[3].results
res = approximate(game; n=20, N=50000, p=2, iterations=10, hook)
p = density(results[1].ξ);
for i in eachindex(results)[2:end]
    results[i].ξ
end
p