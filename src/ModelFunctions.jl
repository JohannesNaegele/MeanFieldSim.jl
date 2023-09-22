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
            x += wₖ * game.C₂ * 1 / α(k * h, game.γ, h, ε[1:k, i, 2]) * E[k, i] * v[k](Float32.(vec(ε[1:k, i, :]))) # TODO: check dimension with NN
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