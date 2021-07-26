from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland_scratch.flatland_utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation
from flatland.envs.agent_utils import RailAgentStatus

from sklearn.neighbors import KDTree

from .rvnn import tree_obs_expand

import numpy as np
import PIL
import math

# GENERAL UTILS=================================================================
def get_agent_target_distance(env, handle):
    """Get agent's distance to target ALONG rail."""
    distance_map = env.distance_map.get()
    agent = env.agents[handle]

    position, direction = agent.position, agent.direction

    if position is None:
        position = agent.initial_position
    if direction is None:
        direction = agent.initial_direction

    return distance_map[handle, position[0], position[1], direction]


def get_agent_positions(env): # Ignores DONE_REMOVED agents
    """Get agent (x, y) positions, ignoring DONE_REMOVED agents"""
    out, agent_handles = [], []

    for i, agent in enumerate(env.agents):
        if agent.status == RailAgentStatus.DONE_REMOVED:
            continue

        if agent.position is None:
            out.append(agent.initial_position)
        else:
            out.append(agent.position)

        agent_handles.append(i)

    return out, agent_handles # Positions and their associated agent handles


def get_valid_actions(env, handle):
    """Get valid actions, remapped from valid transitions in map."""
    agent = env.agents[handle]

    if agent.position is None:
        position, direction = agent.initial_position, agent.direction
    else:
        position, direction = agent.position, agent.direction

    # Get valid transitions and map to actions
    transitions = env.rail.get_transitions(*position, direction)
    valid_actions = set([
        (i + 2 - direction) % 4 for i, x in enumerate(transitions) if x
    ])

    valid_actions.add(4)
    valid_actions.add(0)

    return valid_actions


def get_num_agents_on_map(env):
    return max(1, sum([1 for agent in env.agents if agent.status == RailAgentStatus.ACTIVE]))


def semi_normalise_node(env, node, num_agents_on_map):
    if len(node) == 13:
        node = list(node)[:-1]
    else:
        node = list(node)

    # Normalise
    if num_agents_on_map == 0:
        num_agents_on_map = 1

    node[7] = node[7] / num_agents_on_map
    node[8] = node[8] / num_agents_on_map
    node[11] = node[11] / env.number_of_agents

    for i in [0, 1, 2, 3, 4, 6]:
        if math.isinf(node[i]):
            node[i] = 999

    return node

def semi_normalise_tree_obs(env, obs, handle, num_agents_on_map, degree=2):
    """Do normalisation on EXPANDED tree observation node."""
    node = obs[handle]
    nodes = []

    for child_node in tree_obs_expand(node, degree=degree):
        nodes.extend(semi_normalise_node(env, child_node, num_agents_on_map))

    return nodes


# ROOT EXTRA ===================================================================
def get_self_extra_states(env, obs, handle):
    agent = env.agents[handle]

    # Get valid actions
    action_state = [999, 999, 999, 999]
    for idx, action in enumerate(['B', 'L', 'F', 'R']):
        child_node = obs[handle].childs[action]

        if type(child_node) is float:
            continue

        if math.isinf(child_node.dist_min_to_target):
            continue

        action_state[idx] = child_node.dist_min_to_target

    # Get agent directions
    direction_state = [0, 0, 0, 0]
    direction = agent.direction
    if direction is None:
        direction = agent.initial_direction

    direction_state[direction] = 1

    # Get agent status
    status_state = [0, 0, 0, 0]
    status_state[agent.status] = 1

    # Populate state vector
    state = [
        *action_state, # 4-vector, [F, L, R, B] : min dist to target,
        *direction_state, # 4-vector (1 hot), [N, S, W, E]
        *status_state, # 4-vector (1 hot)
    ]
    return state


def get_self_extra_knn_states(env, handle,
                              agent_handles, kd_tree,
                              k_num=5):
    # +1 because an agent will likely find itself to be closest
    k = min(len(kd_tree.get_arrays()[0]), k_num + 1)

    other_agent_distances = [get_agent_target_distance(env, handle) for handle in env.get_agent_handles()]

    try:
        max_min_target_dist = max(
            dist for dist in other_agent_distances if not (math.isnan(dist) or math.isinf(dist))
        )
    except:
        max_min_target_dist = 999

    agent = env.agents[handle]

    # other_status (1 hot):
    #  READY_TO_DEPART
    #  ACTIVE
    #  DONE
    #  DONE_REMOVED
    # other_malfunction_timer
    # other_speed
    # other_x - agent_x
    # other_y - agent_y
    # 1 - other_min_dist_to_target / max_min_dist_to_target_amongst_agents
    #   ::Note: 0 here means 'furthest away from target' => lowest priority
    k_n_agent_state = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(k_num)]

    agent_pos = agent.position
    if agent_pos is None:
        agent_pos = agent.initial_position

    dist, ind = kd_tree.query([agent_pos], k=k)

    # We need to remap handles since we skip DONE_REMOVED agents
    # NOTE: This uses numpy fancy indexing!!
    other_handles = np.array(agent_handles)[ind][0, :]
    other_positions = kd_tree.get_arrays()[0][ind][0, :]

    found_self = 0

    for kth_nearest_idx, other in enumerate(zip(other_handles, other_positions)):
        other_handle, other_pos = other

        ## HANDLE FINDING SELF
        if other_handle == handle:
            found_self += 1
            continue

        # Shift indices if self has been found
        kth_nearest_idx -= found_self

        ## COMPUTE STATE VECTOR COMPONENTS FOR K NEAREST AGENTS
        other_status = [0, 0, 0, 0]
        other_status[env.agents[other_handle].status] = 1

        if max_min_target_dist == 0:
            soft_priority = 0
        else:
            soft_priority = (
                1 - (get_agent_target_distance(env, other_handle.squeeze())
                / max_min_target_dist)
            )

        # Remove invalid values
        if math.isnan(soft_priority) or math.isinf(soft_priority):
            soft_priority = 0

        # Populate state vector for kth nearest agent
        try: # Just in case it over traverses the list
            k_n_agent_state[kth_nearest_idx] = [
                *other_status, # unpack list
                env.agents[other_handle].malfunction_data['malfunction'],
                env.agents[other_handle].speed_data['speed'],
                other_pos[0] - agent_pos[0], # x distance
                other_pos[1] - agent_pos[1], # y distance
                soft_priority,
            ]
        except:
            pass

    # Unpack the state vector
    k_n_agent_state_vector = [item for sublist in k_n_agent_state for item in sublist]
    return k_n_agent_state_vector
