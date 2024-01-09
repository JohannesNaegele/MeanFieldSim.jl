using MeanFieldSim
using Flux
using JLD2

# TODO: adapt push! aka setindex
(h::ResultsHook)(::PostIterationStage, game, vars; kwargs...) = push!(
    h.results,
    (ξ=deepcopy(vars.ξ), ψ=deepcopy(vars.ψ), ψ_repr=deepcopy(vars.ψ_repr))
)

# Logging
hook = ComposedHook([TimeCostPerTraining(), PrintNet(), ResultsHook(), ValidationHook(20)])

# Game setup
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

# Start simulation
approximate(game; n=20, N=50000, p=2, iterations=10, hook=hook)
# Save results in variable
first_experiment = hook[3].results

# TODO: adapt push! aka setindex
(h::ResultsHook)(::PostIterationStage, game, vars; kwargs...) = nothing
(h::ResultsHook)(::PostExperimentStage, game, vars; kwargs...) = push!(
    h.results,
    (ξ=deepcopy(vars.ξ), ψ=deepcopy(vars.ψ), ψ_repr=deepcopy(vars.ψ_repr))
)

# New variable to also save results
second_experiment = []

lambdas = [0.0, 0.2, 0.4]
gammas = [0.15, 0.3, 0.45]
rhos = [0.0, 0.25, 0.5, 0.75]

# different λ
hook = ResultsHook()
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
    approximate(game; n=20, N=50000, p=2, iterations=10, hook=ComposedHook([PrintNet(), hook]))
end
push!(second_experiment, hook.results)

# different γ
hook = ResultsHook()
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
    approximate(game; n=20, N=50000, p=2, iterations=10, hook=ComposedHook([PrintNet(), hook]))
end
push!(second_experiment, hook.results)

hook = ResultsHook()
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
    approximate(game; n=20, N=50000, p=2, iterations=10, hook=ComposedHook([PrintNet(), hook]))
end
push!(second_experiment, hook.results)

# Save results
jldsave("results.jld2"; first_experiment, second_experiment, lambdas, gammas, rhos)