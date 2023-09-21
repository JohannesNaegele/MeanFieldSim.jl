abstract type AbstractHook end
abstract type AbstractStage end

function (hook::AbstractHook)(::AbstractStage, policy, env) end

struct PreExperimentStage <: AbstractStage end
struct PostExperimentStage <: AbstractStage end
struct PreEpisodeStage <: AbstractStage end
struct PostEpisodeStage <: AbstractStage end

(h::TimeCostPerEpisode)(::PreEpisodeStage, game::MeanFieldGame, vars::SimulatedVariables) = h.t = time_ns()