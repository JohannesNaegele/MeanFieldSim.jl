module MeanFieldSim

using Flux

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
    return exp(γ * sqrt(h) * sum(ε) - γ^2*t/2)
end

function update_value(game, V, E, v, ε, h) # v is a vector of neuronal networks
    V .= game.V₀ * E[1, :]
    for i in eachindex(V)
        x = 0.0
        for k in axes(E, 2)
            wₖ = (k == 1) || (k == size(E)[1]) ? 1.0/2.0 : 1.0
            x += wₖ * game.C₂^2 * 1/α(k*h, γ, h, ε[1:k, i, 2]) * E[k, i] * v[k](ε[1:k, i, :]) # TODO: check dimension with NN
        end
        V[i] += h * x
    end
end

function update_η(game, η, h, ε)
    for i in eachindex(η)
        η[i] = exp(-game.γ_star * V[i] + game.ρ*(game.λ * sqrt(h) * sum(ε[:, i, 2]) - λ^2 * game.T/2))
    end
    η[i] ./= sum(η)
end

function update_sdf(ξ, α, η)
    ξ .*= (1 - α)
    ξ .+= α * η
end

# maybe doing this iteratively and using only vector/scalar might be sufficient
# function loss(v, ε::Matrix{T, 2}, y::Matrix{T, 1}) where {T <: Number}
#     x = 0.0
#     for i in axes(ε, 2) # loop over N columns
#         x += (v(ε[:, i]) - y[i])^2
#     end
#     return x
# end

function approximate(game::MeanFieldGame; n=20, N=50000, p=2, iterations=100, epochs=3)
    # FIXME: Summen k=0???
    ε = randn(n, N, 2)
    E = Matrix{Float64}(undef, n, N)

    h = game.T / n

    for i in axes(E, 1)
        for k in axes(E, 2)
            E[k, i] = exp(game.σ * sum(ε[1:k, i, 1]) + (game.μ - game.σ^2/2)* h * k)
        end
    end

    ξ = ones(N)
    V = Vector{Float64}
    η = Vector{Float64}
    # https://stats.stackexchange.com/a/136542/297734
    n_hidden(n_data, in, α=3.0) = Int(round(n_data/(α * (1 + in))/32)*32)

    optimizer = Adam()
    v = [
            Chain(
                Dense(k => n_hidden(N, 2*n), leakyrelu),   # activation function inside layer
                Dense(n_hidden(N, 2*n) => 1),
                Dropout(0.5),
                leakyrelu
            )
            for k in 1:n
        ]

    for q in 1:iterations
        step_size = 2/(p + q)

        # mehrere epochen, mache backpropagation auf batch aus N samples
        for k in eachindex(v)
            data = [ε[k, :, :], E[k, :] .* ξ[i]] # very inperformant
            for _ in 1:epochs
                model = Flux.setup(v[i], optimizer)
                Flux.train!((m,x,y) -> (m(x) - y)^2, model, data, optim)
            end
        end
        # train schritte geben uns vₖ

        update_value(game, V, E, v, ε, h)
        update_η(game, η, h, ε)
        update_sdf(ξ, step_size, η)
    end
    return ξ
end

end # module MeanFieldGame