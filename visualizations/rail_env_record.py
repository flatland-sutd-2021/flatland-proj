# internal modules
import json
import pickle
from copy import deepcopy

# external modules
from flatland.core.env import Environment
from flatland.envs.rail_env import RailEnv

class RailEnvRecord(Environment):
    def __init__(self, env: RailEnv):
        self.env = env
        self.rail = deepcopy(env.rail)
        self.height = env.height
        self.width = env.width
        self.agents = deepcopy(env.agents)
        self.agents_records = [ self.agents ]
        self.action_records = [ None ]

    def get_record_length(self):
        return len(self.agents_records)

    def set_record_step(self, i):
        self.agents = self.agents_records[i]

    def get_num_agents(self):
        return len(self.agents_records[-1])

    # call after env.step
    def step(self, actions):
        self.agents = deepcopy(self.env.agents)
        self.agents_records.append(self.agents)
        self.action_records.append(actions)

    # clone self and pickle
    def pickle(self, path):
        o = deepcopy(self)
        delattr(o.env.obs_builder, "check_is_observation_valid")
        delattr(o.env.obs_builder, "get_normalized_observation")

        with open(path, "wb") as f:
            pickle.dump(o, f)
