using MeanFieldSim
using Test

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

# (i)
N = 50000
n = 20
ξ = randn(N)
ξ_old = deepcopy(ξ)
step_size = 0.5
η = ones(N)
V = zeros(N)
h = game.T / n

# (ii)


@testset "MeanFieldSim.jl" begin
    MeanFieldSim.update_sdf(ξ, step_size, η)
    @test ξ == 0.5ξ_old .+ 0.5
    # MeanFieldSim.update_η(game, V, η, h, ε)
    # @test η == ...
end