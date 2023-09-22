abstract type AbstractHook end
abstract type AbstractStage end

function (hook::AbstractHook)(::AbstractStage, game, vars; kwargs...) end

# for whole function (especially initialization and saving)
struct PreExperimentStage <: AbstractStage end
struct PostExperimentStage <: AbstractStage end
# for every q iteration
struct PreIterationStage <: AbstractStage end
struct PostIterationStage <: AbstractStage end
# for whole training over all k
struct PreTrainingStage <: AbstractStage end
struct PostTrainingStage <: AbstractStage end
# for training single neuronal networks (beware of threading)
struct PreNetStage <: AbstractStage end
struct PostNetStage <: AbstractStage end
# for each episode of training (beware of threading)
struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end

struct ComposedHook <: AbstractHook
    hooks::Vector{AbstractHook}
end

# make indexable
function Base.getindex(h::ComposedHook, i::Int)
    return h.hooks[i]
end

# ComposedHook(args...) = ComposedHook(args) # slurping allowed

function (hook::ComposedHook)(stage::AbstractStage, game, vars; kwargs...)
    for h in hook.hooks
       h(stage, game, vars; kwargs...)
    end
end

Base.@kwdef struct ResultsHook <: AbstractHook
    results=[]
end

Base.@kwdef mutable struct TimeCostPerTraining <: AbstractHook
    t::UInt64 = time_ns()
    time_costs::Vector{UInt64} = []
end

Base.@kwdef mutable struct PrintNet <: AbstractHook
end

# Base.@kwdef mutable struct IntermediateResult <: AbstractHook
#     game::MeanSimGame
#     simulation_results = []
# end

(h::TimeCostPerTraining)(::PreTrainingStage, game, vars) = h.t
(h::TimeCostPerTraining)(::PostTrainingStage, game, vars) = push!(h.time_costs, time_ns() - h.t)

function (h::PrintNet)(
    ::PreNetStage,
    game::MeanFieldGame,
    vars::SimulatedVariables;
    q,
    k,
    kwargs...
)
    println("Neuronal Network with q: $q, k: $k")
end

struct EmptyHook <: AbstractHook
end

# (h::IntermediateResult)(::PreExperimentStage, game, vars) = push!(h.time_costs, time_ns() - h.t)

struct ValidationHook{T} <: AbstractHook
    training_error::Vector{T}
    test_error::Vector{T}
end

ValidationHook(n::Int) = ValidationHook([0.0 for i in 1:n], [0.0 for i in 1:n])

function (h::ValidationHook)(::PostNetStage, game, vars; k, test_index, kwargs...)
    # for k in eachindex(game.v)
        # x = 0.0
        # for i in game.dings[begin:test_index - 1]
        #     x += game.v[k]()
        # end
        # x /= (test_index - 1)

        # x = 0.0
        # for i in game.dings[test_index:end]
        #     x += game.v[k]()
        # end
        # x /= (test_index - 1)

        # TODO: handle missing tests
        x = [Float32.(vec(vars.ε[1:k, i, :])) for i in eachindex(vars.ξ)]
        y = [Float32(vars.E[k, i] * vars.ξ[i]) for i in eachindex(vars.ξ)]
        # data = [(Float32.(vec(vars.ε[1:k, i, :])), Float32(vars.E[k, i] * vars.ξ[i])) for i in eachindex(vars.ξ)]
        function loss(m, x, y)
            z = 0.0
            for i in eachindex(x)
                z += (m(x[i]) - y[i])^2
            end
            return z/length(x)
        end
        loss_train = loss(vars.v[k], x[begin:test_index - 1], y[begin:test_index - 1])
        loss_test = loss(vars.v[k], x[test_index:end], y[test_index:end])
        h.training_error[k] = loss_train
        h.test_error[k] = loss_test
    # end
end

function (h::ValidationHook)(::PostTrainingStage, game, vars; kwargs...)
    println()
end