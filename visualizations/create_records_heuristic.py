# builtin modules
import os
import sys
from copy import deepcopy

# internal modules
from vis_utils import STARTER_PATH, ENV_RECORDS_PATH
from vis_utils import create_params, create_env, load_context
from vis_eval import eval_policy_visual_heuristic, generate_gif
from vis_eval import DeadLockAvoidanceAgent, get_action_size

AGENT_COUNTS = [1, 2, 5, 7, 9]
AGENT_TYPE = DeadLockAvoidanceAgent

for n_agents in AGENT_COUNTS:
    print("FOR AGENT COUNT:", n_agents)

    path = f'visualizations/context/agents_{n_agents}_hidsize_{256}.context.pickle'
    env, tree_obs, env_params, obs_params, train_params = load_context(path)

    filename = f'label_{AGENT_TYPE.__name__}_agents_{n_agents}.envrecord.pickle'
    if filename in os.listdir(ENV_RECORDS_PATH):
        print("env_record already generated. skipping")
        continue

    policy = AGENT_TYPE(env, get_action_size(), False)

    try:
        record_env, score, completion, step = eval_policy_visual_heuristic(
            env,
            tree_obs,
            policy,
            train_params,
            obs_params,
        )
    except Exception as e:
        print(e)
        continue

    record_env.pickle(filename)
