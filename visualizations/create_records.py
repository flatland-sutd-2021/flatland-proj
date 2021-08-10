# builtin modules
import os
from copy import deepcopy

# internal modules
from vis_utils import STARTER_PATH, ENV_RECORDS_PATH
from vis_utils import create_params, create_env, create_policy, load_context
from vis_eval import eval_policy_visual, generate_gif
from checkpoints import checkpoints


AGENT_COUNTS = [1, 2, 5, 7, 9]
CHECKPOINTS_PATH = "/reinforcement_learning/checkpoints/"

for n_agents in AGENT_COUNTS:
    print("FOR AGENT COUNT:", n_agents)
    for label, meta in checkpoints.items():
        # get state size and checkpoint path and trained episode
        eps = meta["trained_episodes"]
        hidsize = meta["hidden_size"]
        state_size = meta["state_size"]
        checkpoint = os.path.normpath(
            STARTER_PATH
            + CHECKPOINTS_PATH
            + f'/{meta["path"]}/{meta["training_id"]}-{meta["trained_episodes"]}.pth'
        )

        # create the same environment if the context is the same
        path = f'visualizations/context/agents_{n_agents}_hidsize_{hidsize}.context.pickle'
        env, tree_obs, env_params, obs_params, train_params = load_context(path)

        filename = f'eps_{eps}_label_{label}_agents_{n_agents}.envrecord.pickle'
        if filename in os.listdir(ENV_RECORDS_PATH):
            print("env_record already generated. skipping")
            continue

        policy = create_policy(env, state_size, train_params, checkpoint)

        try:
            record_env, score, completion, step = eval_policy_visual(
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
