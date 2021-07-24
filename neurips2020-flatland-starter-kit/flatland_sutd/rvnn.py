import torch.nn.functional as F
import torch.nn as nn
import torch

from collections import deque
import copy
import math


# BFS NODES ====================================================================
def tree_obs_expand(obs_node, degree=2):
    n_inf = float('-inf')

    if type(obs_node) is float:
        if math.isinf(obs_node):
            return [float('-inf')] * degree

    children = []

    for action, child in obs_node.childs.items():
        if type(child) is float:
            children.append((float('inf'), float('inf')))
        else:
            children.append((child.dist_min_to_target, child))

    children.sort(key=lambda x:x[0])

    out = []
    last_child = float("-inf")
    for idx, child in zip(range(degree), children): # Works on sorted
        if type(child[1]) is not float:
            last_child = child[1] # Duplicate children if missing
        out.append(last_child)

    return out


def leaf_to_root(root, expand_fn=None, depth_limit=4):
    """Construct dict of children at depths from a tree."""
    depth_queues = {0: deque([root])}
    visited = [root]
    queue = [(root, 0)]

    while queue:
        node, depth = queue.pop(0)

        if expand_fn is None:
            children = node.children
        else:
            children = expand_fn(node)

        for child in children:
            # if child not in visited:
            child_depth = depth + 1
            if child_depth >=depth_limit:
                continue

            if depth_queues.get(child_depth) is None:
                depth_queues[child_depth] = deque([child])
            else:
                depth_queues[child_depth].append(child)

            visited.append(child)
            queue.append((child, child_depth))

    return depth_queues


def depth_queues_reduce(depth_queues, degree, reduce_fn):
    """Apply reduction operation recursively over a depth_queue dict."""
    queue_list = copy.deepcopy(list(depth_queues.items()))

    while len(queue_list) > 2:
        child_depth, children = queue_list[-1]
        parent_depth, parents = queue_list[-2]

        while children:
            child_list = []
            for _ in range(degree):
                child_list.append(children.pop())
            parent = parents.pop()
            parents.appendleft(reduce_fn(child_list, parent))
        queue_list.pop()

    return queue_list


# RVNN =========================================================================
# Recursive Neural Network
class RvNN(nn.Module):
    def __init__(self, children_states_size, parent_state_size, output_size, expand_fn, device="cuda", prelatent_size=128, latent_size=256, max_depth=4):
        super(RvNN, self).__init__()
        self.device = device
        self.max_depth = max_depth
        self.expand_fn = expand_fn

        self.child_encoder = nn.Sequential(
            nn.Linear(children_states_size, prelatent_size),
            nn.PReLU(),
            nn.Linear(prelatent_size, latent_size),
            nn.Tanh(),
        ).to(self.device)

        self.parent_encoder = nn.Sequential(
            nn.Linear(parent_state_size, prelatent_size),
            nn.PReLU(),
            nn.Linear(prelatent_size, latent_size),
            nn.Tanh(),
        ).to(self.device)

        self.latent_output = nn.Sequential(
            nn.Linear(2*latent_size, prelatent_size),
            nn.Tanh(),
            nn.Linear(prelatent_size, output_size),
            nn.PReLU(),
        ).to(self.device)

    def forward(self, root_node):
        """Recursive forward!! Pass in the root node of your tree"""
        depth_queues = leaf_to_root(root_node, self.expand_fn, depth_limit=self.max_depth)

        # Strategy: Apply reduction operation recursively, popping from lowest depth
        reduced = depth_queues_reduce(depth_queues, 2, self._forward_step)
        return torch.cat(tuple(reduced[1][1]))

    def _forward_step(self, children, parent):
        children_states = []

        try:
            parent_states = list(parent[:12])
        except:
            parent_states = [0] * 12

        for child in children:
            try:
                children_states.extend(child[:12])
            except:
                children_states.extend([0] * 12)

        for idx, val in enumerate(children_states):
            if math.isinf(val):
                children_states[idx] = 9999
            if math.isnan(val):
                children_states[idx] = 0

        for idx, val in enumerate(parent_states):
            if math.isinf(val):
                parent_states[idx] = 9999
            if math.isnan(val):
                parent_states[idx] = 0

        children_out = self.child_encoder(torch.FloatTensor(children_states).to(self.device))
        parent_out = self.parent_encoder(torch.FloatTensor(parent_states).to(self.device))

        latent = torch.cat((children_out, parent_out))
        return self.latent_output(latent)


# RECURSIVE TREE REDUCTION =====================================================
if __name__ == "__main__":
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import sparse_rail_generator
    from flatland.envs.schedule_generators import sparse_schedule_generator
    from flatland_scratch.flatland_utils.observation_utils import normalize_observation
    from flatland.envs.observations import TreeObsForRailEnv

    MAX_DEPTH = 4

    n_agents = 5
    x_dim = 25
    y_dim = 25
    n_cities = 4
    max_rails_between_cities = 2 # Max rails connecting cities
    max_rails_in_city = 3 # Max rails inside city (num tracks in train stations)
    seed = 42
    grid_mode = True # Distribute cities in a grid vs randomly

    obs = TreeObsForRailEnv(max_depth=MAX_DEPTH)
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            seed=seed,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        obs_builder_object=obs
    )
    env_obs, info = env.reset()

    top = env_obs[0]

    depth_queues = leaf_to_root(top, tree_obs_expand, depth_limit=MAX_DEPTH)
    depth_queues

    def rvnn_reduce(children, parent):
        children_states = []

        for child in children:
            children_states.extend(child[:13])

        return RvNN(children, parent[:13])

    # def sum_reduce(children, parent):
    #     """Example reduction function: Simple summing"""
    #     return Node(-1, data=sum([child.data for child in children], parent.data))

    # Strategy: Apply reduction operation recursively, popping from lowest depth
    reduced = depth_queues_reduce(depth_queues, 2, rvnn_reduce)

    depth_queues
    leaf_to_root(depth_queues[1][0]) # Depth queue for first child
    reduced
