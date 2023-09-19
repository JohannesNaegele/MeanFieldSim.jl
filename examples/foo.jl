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

ξ = approximate(game; n=20, N=50000, p=2, iterations=10)
density(ξ)

c = Chain(
    Dense(10 => 5, leakyrelu),   # activation function inside layer
    Dense(5 => 1),
    Dropout(0.5),
    leakyrelu,
    only
)
# data = [(c(hcat(rand(10), rand(10))), hcat([1], [1]))]
data = [([x*i for i in 1:10], x^2) for x in 1:10]

Flux.train!((m,x,y) -> (m(x) - y)^2, c, data, opt)

opt = Flux.setup(Adam(), c)