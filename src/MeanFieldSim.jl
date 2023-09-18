module MeanFieldSim

struct MeanFieldGame{T}
    c::T # emission efficiacy
    γ::T # risk aversion parameter
    ρ::T # proportion of green investors
    T::T # time horizon
    λ::T # environmental concern
    μ::T # drift of value
    σ::T # volatility of common risk
    V₀::T # average initial firm value
end

"""
This implements sampling from a n-dimensional brownian motion.
"""
B(n, t) = ...

# """
# samples ?
# """
# function α(t)
#     return exp(γ*B(2, t) - γ^2*t/2)
# end

function α(t, γ, h, ε)
    return exp(γ * sqrt(h) * sum(ε) - γ^2*t/2)
end

# maybe doing this iteratively and using only vector/scalar might be sufficient
function loss(v, ε::Matrix{T, 2}, y::Matrix{T, 1}) where {T <: Number}
    x = 0.0
    for i in axes(ε, 2) # loop over N columns
        x += (v(ε[:, i]) - y[i])^2
    end
    return x
end

function update_value(game, V, E, v, ε) # v is a vector of neuronal networks
    h = game.T/n
    V .= game.V₀ * E[1, :]
    for i in eachindex(V)
        x = 0.0
        for k in axes(E, 2)
            wₖ = (k == 1) || (k == size(E)[2]) ? 1.0/2.0 : 1
            x += wₖ * game.c^2 * 1/α(t, k*h, h, ε[:, i, 2]) * E[k, i] * v[k](E[1:k, i])
        end
        V[i] += h * x
    end
end

function update_η(game, η, h, ε)
    for i in eachindex(η)
        η[i] = exp(-game.γ * V + game.ρ*(game.λ * sqrt(h) * sum(ε[:, i, 1]) - λ^2 * T/2))
    end
    η[i] ./= sum(η)
end

function update_sdf(ξ, α, η)
    ξ .*= (1 - α)
    η .+= α * η
end

function approximate(game::MeanFieldGame; n=100, k=0.1, N=100, p=10, iterations=100)
    ε = Array{Float64}(undef, n, N, 2)
    E = Matrix{Float64}(undef, n, N)
    ξ = ones(N)
    V = Vector{Float64}
    η = Vector{Float64}

    for q in 1:iterations
        step_size = 2/(p + q)


        # mehrere epochen, mache backpropagation auf batch aus N samples
        # train schritte geben uns vₖ
        update_value()
        update_η()
        update_sdf(ξ, step_size, η)
    end
end

end # module MeanFieldGame
