module MeanFieldSim

struct MeanFieldGame

c::Float64
γ::Float64

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

end
