module MeanFieldSim

struct MeanFieldGame
    c::Float64
    γ::Float64
    ρ::Float64
    T::Float64
    λ::Float64
end

"""
This implements sampling from a n-dimensional brownian motion.
"""
B(n, t) = ...

"""
samples ?
"""
function α(t)
    return exp(γ*B(2, t) - γ^2*t/2)
end

# maybe doing this iteratively and using only vector/scalar might be sufficient
function loss(v, ε::Matrix{T, 2}, y::Matrix{T, 1}) where {T <: Number}
    x = 0.0
    for i in axes(ε, 2) # loop over N columns
        x += (v(ε[:, i]) - y[i])^2
    end
    return x
end

function update_sdf(ξ, α, η)
    ξ .*= (1 - α)
    η .+= α * η
end

function approximate(game::MeanFieldGame; N=100; p=10, iterations=100)
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
