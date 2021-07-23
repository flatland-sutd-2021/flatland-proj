from typing import Dict

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.env_prediction_builder import PredictionBuilder

def get_agent_priority_naive(
    env: RailEnv, predictor: PredictionBuilder
) -> Dict[int, float]:
    """
    Gets priority values (normalized from 0 to 1) for the agents using the naive method
    (agents collide if predictor predicts that they are are the same tile at the same time)
    """

    # set env if it is not already set
    predictor.set_env(env)

    # set up conflict graph
    conflict_graph = {i: set() for i in range(env.number_of_agents)}

    # predictions for the location of the agents
    # predictions[i] is an np array with shape (max_depth + 1, 3)
    # where the columns correspond to (time_offset, positon_axis_0, position_axis_1)
    predictions = {i: predictor.get(i)[i][:, :3] for i in range(env.number_of_agents)}

    # construct conflict graph
    for i in range(env.number_of_agents):
        for j in range(env.number_of_agents):
            # ignore if the agents are the same
            if i == j:
                continue

            # ignore if either agent is not active
            if (
                env.agents[i].status != RailAgentStatus.ACTIVE
                or env.agents[i].status != RailAgentStatus.ACTIVE
            ):
                continue

            # if agent[i] and agent[j] are in the same predicted position at the same time
            for t in range(predictor.max_depth):
                if (predictions[i][t] == predictions[j][t]).all():
                    conflict_graph[i].add(j)
                    conflict_graph[j].add(i)
                    break

    # priority = degree of conflict graph node
    priorities = {i: len(conflict_graph[i]) for i in range(env.number_of_agents)}

    # get largest priority degree
    max_degree = max([d for _, d in priorities.items()])

    # set max degree to minimum value of 1 to preven division by zero error
    max_degree = max(max_degree, 1)

    # normalize priority values
    normalized_priorities = {p: (d / max_degree) for p, d in priorities.items()}

    return normalized_priorities

if __name__ == "__main__":
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv
    from flatland.envs.schedule_generators import sparse_schedule_generator
    from flatland.envs.rail_generators import sparse_rail_generator

    predictor = ShortestPathPredictorForRailEnv()
    obs = TreeObsForRailEnv(max_depth=2)
    env = RailEnv(
        width=25,
        height=25,
        rail_generator=sparse_rail_generator(
            max_num_cities=4,
            seed=42,
            grid_mode=True,
            max_rails_between_cities=2,
            max_rails_in_city=3
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=10,
        obs_builder_object=obs,
    )

    env.reset()

    # step all 10 agents forward 10 times
    for _ in range(10):
        env.step({ i : 2 for i in range(10) })

    agent_priorities = get_agent_priority_naive(env, predictor)
    print(agent_priorities)
