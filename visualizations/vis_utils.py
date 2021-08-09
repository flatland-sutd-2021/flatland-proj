# builtin modules
import sys
import os
import pickle
from argparse import Namespace

# external modules
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

# resolve internal imports
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CONTEXT_PATH = os.path.join(DIR_PATH, "context")
ENV_RECORDS_PATH = os.path.join(DIR_PATH, "env_records")
STARTER_PATH = os.path.join(DIR_PATH, "..", "neurips2020-flatland-starter-kit")
sys.path.append(os.path.normpath(STARTER_PATH))

# internal modules
from reinforcement_learning import multi_agent_training
from utils.observation_utils import normalize_observation
from utils.agent_action_config import get_action_size
from reinforcement_learning.ppo_agent import PPOPolicy
from reinforcement_learning.deadlockavoidance_with_decision_agent import (
    DeadLockAvoidanceWithDecisionAgent,
)


def create_params(n_agents=2):
    env_params = {
        "n_agents": n_agents,
        "x_dim": 30,
        "y_dim": 30,
        "n_cities": 2,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 3,
        "malfunction_rate": 1 / 100,
        "seed": 0,
    }

    obs_params = {
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 10,
    }

    train_params = {
        "n_episodes": 5000,
        "n_agent_fixed": False,
        "training_env_config": 1,
        "evaluation_env_config": 1,
        "n_evaluation_episodes": 10,
        "checkpoint_interval": 100,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.9975,
        "buffer_size": 32000,
        "buffer_min_size": 0,
        "restore_replay_buffer": "",
        "save_replay_buffer": False,
        "batch_size": 128,
        "gamma": 0.97,
        "tau": 0.0005,
        "learning_rate": 5e-05,
        "hidden_size": 256,
        "update_every": 10,
        "use_gpu": False,
        "num_threads": 4,
        "render": False,
        "load_policy": "",
        "use_fast_tree_observation": False,
        "max_depth": 2,
        "policy": "DeadLockAvoidance",
        "action_size": "full",
    }

    env_params = Namespace(**env_params)
    obs_params = Namespace(**obs_params)
    train_params = Namespace(**train_params)

    return env_params, obs_params, train_params


def create_env(env_params, obs_params):
    predictor = ShortestPathPredictorForRailEnv(obs_params.observation_max_path_depth)

    tree_observation = TreeObsForRailEnv(
        max_depth=obs_params.observation_max_path_depth,
        predictor=predictor,
    )

    def check_is_observation_valid(observation):
        return observation != None

    def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
        return normalize_observation(observation, tree_depth, observation_radius)

    tree_observation.check_is_observation_valid = check_is_observation_valid
    tree_observation.get_normalized_observation = get_normalized_observation

    env = multi_agent_training.create_rail_env(
        env_params,
        tree_observation,
    )

    env.reset(regenerate_rail=True, regenerate_schedule=True)

    return env, tree_observation, obs_params


def create_policy(env, state_size, train_params, checkpoint):
    inter_policy = PPOPolicy(
        state_size,
        get_action_size(),
        use_replay_buffer=False,
        in_parameters=train_params,
    )

    policy = DeadLockAvoidanceWithDecisionAgent(
        env,
        state_size,
        get_action_size(),
        inter_policy,
    )

    policy.load(checkpoint)

    return policy


def load_context(path):
    with open(path, "rb") as f:
        c = pickle.load(f)

    def check_is_observation_valid(observation):
        return observation != None

    def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
        return normalize_observation(observation, tree_depth, observation_radius)

    c["tree_obs"].check_is_observation_valid = check_is_observation_valid
    c["tree_obs"].get_normalized_observation = get_normalized_observation

    return c["env"], c["tree_obs"], c["env_params"], c["obs_params"], c["train_params"]


def create_and_store_context(n_agents):
    env_params, obs_params, train_params = create_params(n_agents=n_agents)
    env, tree_observation, obs_params = create_env(env_params, obs_params)

    delattr(tree_observation, "check_is_observation_valid")
    delattr(tree_observation, "get_normalized_observation")

    o = {
        "env": env,
        "tree_obs": tree_observation,
        "env_params": env_params,
        "obs_params": obs_params,
        "train_params": train_params,
    }

    path = f"agents_{n_agents}.context.pickle"

    with open(path, "wb") as f:
        pickle.dump(o, f)


def load_env_record(path):
    with open(path, "rb") as f:
        env_record = pickle.load(f)

    return env_record
