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
    k
)
    println("Neuronal Network with q: $q, k: $k")
end

struct EmptyHook <: AbstractHook
end

# (h::IntermediateResult)(::PreExperimentStage, game, vars) = push!(h.time_costs, time_ns() - h.t)
