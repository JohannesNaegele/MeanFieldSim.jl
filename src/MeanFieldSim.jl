module MeanFieldSim

export MeanFieldGame, SimulatedVariables
export AbstractHook, AbstractStage
export
    PreExperimentStage,
    PostExperimentStage,
    PreIterationStage,
    PostIterationStage,
    PreTrainingStage,
    PostTrainingStage
    # PreNetStage,
    # PostNetStage
export ComposedHook, ResultsHook, ValidationHook, TimeCostPerTraining, PrintNet
export approximate

using Statistics
using Flux
import Flux.Data: DataLoader

Base.@kwdef struct MeanFieldGame{S<:Any}
    C::S # emission efficiacy
    C₂::S # emission efficiacy ???
    γ::S # risk aversion parameter
    γ_star::S # risk aversion parameter ???
    ρ::S # proportion of green investors
    T::S # time horizon
    λ::S # environmental concern
    μ::S # drift of value
    σ::S # volatility of common risk
    V₀::S # average initial firm value
end

Base.@kwdef struct SimulatedVariables # TODO: parametrise
    ε
    E
    ξ
    ψ
    ψ_repr
    V
    η
    # optimizer
    v
    # batch_size
end

include("hooks.jl")
include("ModelFunctions.jl")


function approximate(game::MeanFieldGame; n=20, N=50000, p=2, iterations=10, epochs=1, test_percentage=1/6, hook=EmptyHook())
    # TODO: saving of model
    # FIXME: Starten Summen richtig? (k=0?)

    # FIXME: ugly programming
    # hook = vcat(hook.hooks, )

    ε = randn(n, N, 2)
    E = Matrix{Float64}(undef, n, N)

    h = game.T / n

    for i in axes(E, 2)
        for k in axes(E, 1)
            E[k, i] = exp(game.σ * sum(ε[1:k, i, 1]) + (game.μ - game.σ^2 / 2) * h * k)
        end
    end

    ξ = ones(N) # stochastic discount factor
    ψ = zeros(N)
    ψ_repr = Vector{Float64}(undef, n)
    V = zeros(N)
    η = zeros(N)
    # https://stats.stackexchange.com/a/136542/297734
    n_hidden(n_data, in, α=3.0) = Int(round(n_data / (α * (1 + in)) / 32) * 32)

    optimizer = Adam()
    v = [
        Chain(
            Dense(2k => n_hidden(N, 2 * n, 8.0), leakyrelu),   # activation function inside layer
            Dense(n_hidden(N, 2 * n, 8.0) => 1),
            Dropout(0.5),
            leakyrelu,
            only # important, gives scalar a instead of [a]
        )
        for k in 1:n
    ]
    batch_size = 32

    # loss_fn = (m, x, y) -> mean((m(x) .- y) .^ 2)
    # results = []

    vars = SimulatedVariables(
        ε,
        E,
        ξ,
        ψ,
        ψ_repr,
        V,
        η,
        v
    )

    hook(PreExperimentStage(), game, vars)

    n_test = min(Int(round(N * (1 - test_percentage) / 32)) * 32, N)

    for q in 1:iterations
        step_size = 2 / (p + q)
        hook(PreIterationStage(), game, vars)
        hook(PreTrainingStage(), game, vars)
        # mehrere epochen, mache backpropagation auf batch aus N samples
        Threads.@threads for k in eachindex(v)
            hook(PreNetStage(), game, vars, q=q, k=k)
            # TODO: test data set
            data = [(Float32.(vec(ε[1:k, i, :])), Float32(E[k, i] * ξ[i])) for i in 1:n_test] # very inperformant
            # data_loader = DataLoader(data, batchsize=batch_size, shuffle=true)
            # println("q: $q, k: $k")
            for _ in 1:epochs
                hook(PreEpisodeStage(), game, vars, q=q, k=k)
                opt = Flux.setup(optimizer, v[k])
                # for (x_batch, y_batch) in data_loader
                #     # Flux.Losses.mse
                #     # loss(ŷ, y, agg=x->mean(w .* x))
                #     # TODO: batch size
                #     # (m,x,y) -> mean(m(x) .- y).^2
                #     Flux.train!((m,x,y) -> (m(x) - y)^2, v[k], [(x_batch, y_batch)], opt)
                # end
                loss_function(m, x, y) = (m(x) - y)^2
                Flux.train!(loss_function, v[k], data, opt)
                hook(PostEpisodeStage(), game, vars, q=q, k=k)
            end
            hook(PostNetStage(), game, vars, k=k, test_index=n_test)
        end
        hook(PostTrainingStage(), game, vars)

        # training gibt uns vₖ
        println("update value...")
        update_value(game, V, E, [x -> v[i](x)[1] for i in eachindex(v)], ε, h)
        println("update η...")
        update_η(game, V, η, h, ε)
        println("update stochastic discount function...")
        update_sdf(ξ, step_size, η)
        println("update total average emissions...")
        total_average_emissions!(game, ψ, ε, v, h)
        println("update expected representative emissions...")
        expected_emissions!(game, ψ_repr, ε, E, ξ, h)
        hook(PostIterationStage(), game, vars)
    end
    # return results
    hook(PostExperimentStage(), game, vars)
end

end # module MeanFieldGame