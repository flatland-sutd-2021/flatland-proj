from typing import Dict

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.env_prediction_builder import PredictionBuilder

import numpy as np

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
    predictions = {i: predictor.get(i).get(i)[:, :3] for i in range(env.number_of_agents)}

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
                if np.array_equal(predictions[i][t], predictions[j][t]):
                    conflict_graph[i].add(j)
                    conflict_graph[j].add(i)
                    break

    # priority = degree of conflict graph node
    priorities = np.argsort([len(conflict_graph[i]) for i in range(env.number_of_agents)])
    # priorities = {i: len(conflict_graph[i]) for i in range(env.number_of_agents)}

    # get largest priority degree
    # max_degree = max([d for _, d in priorities.items()])
    max_degree = max(priorities)

    if max_degree == 0:
        max_degree = 1

    # normalize priority values
    normalized_priorities = [(p / max_degree) for p in priorities]

    return normalized_priorities
