abstract type AbstractHook end
abstract type AbstractStage end

function (hook::AbstractHook)(::AbstractStage, policy, env; kwargs...) end

# for whole function (especially initialization and saving)
struct PreExperimentStage <: AbstractStage end
struct PostExperimentStage <: AbstractStage end
# for every q iteration
struct PreIterationStage <: AbstractStage end
struct PostIterationStage <: AbstractStage end
# for whole training over all k
struct PreTrainingStage <: AbstractStage end
struct PostTrainingStage <: AbstractStage end
# for training episodes
struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end

Base.@kwdef mutable struct TimeCostPerTraining <: AbstractHook
    t::UInt64 = time_ns()
    time_costs::Vector{UInt64} = []
end

Base.@kwdef mutable struct PrintEpisode <: AbstractHook
end

# Base.@kwdef mutable struct IntermediateResult <: AbstractHook
#     game::MeanSimGame
#     simulation_results = []
# end

(h::TimeCostPerTraining)(::PreTrainingStage, game, vars) = h.t
(h::TimeCostPerTraining)(::PostTrainingStage, game, vars) = push!(h.time_costs, time_ns() - h.t)

function (h::PrintEpisode)(
    ::PreEpisodeStage,
    game::MeanFieldGame,
    vars::SimulatedVariables;
    q, k
)
    println("Episode with q: $q, k: $k")
end

# (h::IntermediateResult)(::PreExperimentStage, policy, env) = push!(h.time_costs, time_ns() - h.t)
