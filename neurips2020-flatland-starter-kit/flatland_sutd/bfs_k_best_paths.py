from .state_construction import *
from collections import deque
import math

# ==============================================================================
def _get_child_tuples(node_tuple):
    out = []

    node = node_tuple[-1]

    for idx, action in enumerate(['B', 'L', 'F', 'R']):
        child = node.childs.get(action)
        if child == float('-inf') or child is None:
            continue

        out.append(
            (child[5] + child[6], # Total travel distance to target
             node_tuple[1] + 1, # Depth
             node_tuple[2], # Action index
             child)
        )

    return out


def get_k_best_nodes(root, depth_limit=4, k=2, pad_output=True):
    queue = deque()

    # Get all direct children
    for idx, action in enumerate(['B', 'L', 'F', 'R']):
        child = root.childs.get(action)
        if child == float('-inf') or child is None:
            continue

        queue.append(
            (child[5] + child[6], # Total travel distance to target
             1, # Depth
             idx, # Action index
             child)
        )

    # Then do BFS and find k terminal nodes with with the shortest total
    # travel distance to target
    unsorted_terminal_node_tuples = []
    counter = 0

    while queue:
        counter += 1
        node_tuple = queue.popleft()

        # If target found (terminal target node)
        if not math.isinf(node_tuple[-1][0]):
            unsorted_terminal_node_tuples.append(node_tuple)
            continue

        children = _get_child_tuples(node_tuple)

        if len(children) == 0 or node_tuple[1] >= depth_limit: # Terminal node
            unsorted_terminal_node_tuples.append(node_tuple)
            continue

        queue.extend(children)

    # Sort and obtain k_best path nodes
    out = sorted(unsorted_terminal_node_tuples)[:k]
    while len(out) < k and pad_output:
        out.append(out[-1])

    return out


def get_k_best_node_states(root, env, num_agents_on_map,
                           depth_limit=4, k=2, pad_output=True):
    states = []
    node_tuples = get_k_best_nodes(root, depth_limit, k, pad_output)

    for node_tuple in node_tuples:
        node = node_tuple[-1]
        states.extend(semi_normalise_node(env, node, num_agents_on_map))
        states.append(node_tuple[1])

        action_states = [0, 0, 0, 0]
        action_states[node_tuple[2]] = 1
        states.extend(action_states)

    return states
