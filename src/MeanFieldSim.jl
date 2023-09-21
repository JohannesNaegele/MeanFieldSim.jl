module MeanFieldSim

export MeanFieldGame
export approximate

using Statistics
using Flux
import Flux.Data: DataLoader

Base.@kwdef struct MeanFieldGame{S}
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

function α(t, γ, h, ε)
    return exp(γ * sqrt(h) * sum(ε) - γ^2 * t / 2)
end

function update_value(game, V, E, v, ε, h) # v is a vector of neuronal networks
    V .= game.V₀ * E[1, :]
    for i in eachindex(V)
        x = 0.0
        for k in axes(E, 1)
            wₖ = (k == 1) || (k == size(E)[1]) ? 1.0 / 2.0 : 1.0
            # println(v[k](ε[1:k, i, :]))
            x += wₖ * game.C₂^2 * 1 / α(k * h, game.γ, h, ε[1:k, i, 2]) * E[k, i] * v[k](Float32.(vec(ε[1:k, i, :]))) # TODO: check dimension with NN
        end
        V[i] += h * x
    end
end

function update_η(game, V, η, h, ε)
    for i in eachindex(η)
        η[i] = exp(-game.γ_star * V[i] + game.ρ * (game.λ * sqrt(h) * sum(ε[:, i, 2]) - game.λ^2 * game.T / 2))
    end
    η .*= length(η)/sum(η)
end

function update_sdf(ξ, α, η)
    ξ .*= (1 - α)
    ξ .+= α * η
end

function total_average_emissions!(game, ψ, ε, v, h)
    for i in eachindex(ψ)
        ψ[i] = 0.0
        for k in eachindex(v)
            wₖ = (k == 1) || (k == length(v)) ? 1.0 / 2.0 : 1.0
            ψ[i] += wₖ * game.C * 1 / α(k * h, game.γ, h, ε[1:k, i, 2]) * v[k](Float32.(vec(ε[1:k, i, :])))
        end
        ψ[i] *= h
    end
end

# maybe doing this iteratively and using only vector/scalar might be sufficient
# function loss(v, ε::Matrix{T, 2}, y::Matrix{T, 1}) where {T <: Number}
#     x = 0.0
#     for i in axes(ε, 2) # loop over N columns
#         x += (v(ε[:, i]) - y[i])^2
#     end
#     return x
# end

function approximate(game::MeanFieldGame; n=20, N=50000, p=2, iterations=10, epochs=1, intermediate=nothing)
    # TODO: logging, hooks
    # FIXME: Summen k=0???
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
    loss_fn = (m, x, y) -> mean((m(x) .- y) .^ 2)

    results = []

    for q in 1:iterations
        step_size = 2 / (p + q)
        # mehrere epochen, mache backpropagation auf batch aus N samples
        Threads.@threads for k in eachindex(v)
            data = [(Float32.(vec(ε[1:k, i, :])), Float32(E[k, i] * ξ[i])) for i in eachindex(ξ)] # very inperformant
            data_loader = DataLoader(data, batchsize=batch_size, shuffle=true)
            println("q: $q, k: $k")
            for _ in 1:epochs
                opt = Flux.setup(optimizer, v[k])
                # for (x_batch, y_batch) in data_loader
                #     # Flux.Losses.mse
                #     # loss(ŷ, y, agg=x->mean(w .* x))
                #     # TODO: batch size
                #     # (m,x,y) -> mean(m(x) .- y).^2
                #     Flux.train!((m,x,y) -> (m(x) - y)^2, v[k], [(x_batch, y_batch)], opt)
                # end
                Flux.train!((m, x, y) -> (m(x) - y)^2, v[k], data, opt)
            end
        end
        # train schritte geben uns vₖ
        println("update value:")
        update_value(game, V, E, [x -> v[i](x)[1] for i in eachindex(v)], ε, h)
        println("update η:")
        update_η(game, V, η, h, ε)
        println("update stochastic discount function:")
        update_sdf(ξ, step_size, η)
        total_average_emissions!(game, ψ, ε, v, h)
        push!(results, [deepcopy(ξ), deepcopy(ψ)])
    end
    return results
end

end # module MeanFieldGame