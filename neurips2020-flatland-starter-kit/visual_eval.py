from argparse import Namespace
import sys
from typing import List, Dict, Tuple

import imageio
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.core.env_observation_builder import ObservationBuilder

# mac-specific issues
if sys.platform == "darwin":
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

from reinforcement_learning.policy import LearningPolicy
from utils.observation_utils import normalize_observation
from utils.agent_action_config import map_actions
from flatland_sutd import *


def eval_policy_visual(
    env: RailEnv,
    tree_observation: ObservationBuilder,
    policy: LearningPolicy,
    train_params: Namespace,
    obs_params: Namespace,
    renderer: RenderTool,
) -> Tuple[List[np.ndarray], float, float, int]:
    print(env._max_episode_steps)

    print("", flush=True)
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius

    predictor = ShortestPathPredictorForRailEnv(obs_params.observation_max_path_depth)

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    print("EVALUATING POLICY:")

    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    policy.reset(env)

    agent_obs = [None] * env.get_num_agents()
    score = 0.0

    max_steps = env._max_episode_steps
    final_step = 0

    # render initial environment image
    renderer.render_env()
    env_images = [renderer.get_image()]

    policy.start_episode(train=False)

    # create reward modifier for the purposes of tracking stops
    eval_mod = RewardModifier(env)
    for step in range(max_steps - 1):
        print(f"\r step: {step:04}", end="")
        policy.start_step(train=False)

        eval_mod.check_staticness(env)

        # Compute expensive stuff ONCE per step
        if True:  # CH3: Set to True when ready
            agent_positions, agent_handles = get_agent_positions(env)
            kd_tree = KDTree(agent_positions)

            # This is -NOT- total agent count or active agent count!
            num_agents_on_map = get_num_agents_on_map(env)

        for agent in env.get_agent_handles():
            if tree_observation.check_is_observation_valid(agent_obs[agent]):
                agent_obs[agent] = tree_observation.get_normalized_observation(
                    obs[agent],
                    tree_depth=tree_depth,
                    observation_radius=observation_radius,
                )

            action = 0
            if info["action_required"][agent]:
                if True or tree_observation.check_is_observation_valid(
                    agent_obs[agent]
                ):
                    if True:  # CH3: When it is time...
                        rvnn_out = policy.rvnn(obs[agent])
                        state_vector = [
                            # == ROOT ==
                            *get_k_best_node_states(
                                obs[agent],
                                env,
                                num_agents_on_map,
                                obs_params.observation_tree_depth,
                            ),
                            # == ROOT EXTRA ==
                            env.number_of_agents,
                            *get_self_extra_states(env, obs, agent),
                            get_agent_priority_naive(env, predictor)[agent],
                            eval_mod.stop_dict[agent],  # staticness
                            *get_self_extra_knn_states(
                                env, agent, agent_handles, kd_tree, k_num=5
                            ),
                            # == RVNN CHILDREN ==
                            *rvnn_out,
                        ]

                        action = policy.act(agent, state_vector, eps=0.0)
                    else:
                        agent_obs[agent] = tree_observation.get_normalized_observation(
                            obs[agent],
                            tree_depth,
                            observation_radius=observation_radius,
                        )
                        action = policy.act(agent, agent_obs[agent], eps=0.0)

            action_dict.update({agent: action})
        policy.end_step(train=False)

        print(action_dict)
        obs, all_rewards, done, info = env.step(map_actions(action_dict))
        renderer.render_env(show_observations=False)
        env_images.append(renderer.get_image())

        # No reward hacking here

        for agent in env.get_agent_handles():
            score += all_rewards[agent]

        final_step = step

        if done["__all__"]:
            break

    policy.end_episode(train=False)
    normalized_score = score / (max_steps * env.get_num_agents())

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
    completion = tasks_finished / max(1, env.get_num_agents())

    print(
        "\n\n ✅ Eval: score {:.3f} done {:.1f}%\n".format(
            normalized_score, completion * 100.0
        ),
        flush=True,
    )
    policy.report_selector()

    return env_images, normalized_score, completion, final_step


def generate_gif(path: str, env_images: List[np.ndarray]):
    # env_images[i] is of shape (?, ?, 4), with each value in the range 0-254
    frames = []

    for idx, image in enumerate(env_images):
        plt.ioff()
        plt.clf()

        fig = plt.figure()
        ax = fig.subplots()
        ax.set_axis_off()
        ax.set_title(f"Step {idx}")
        ax.imshow(image)

        fig.canvas.draw()

        frame = np.array(fig.canvas.renderer.buffer_rgba()).astype(np.uint8)

        plt.close()

        frames.append(frame)

    imageio.mimwrite(path, frames, fps=10)


if __name__ == "__main__":
    from flatland.envs.rail_generators import sparse_rail_generator
    from flatland.envs.schedule_generators import sparse_schedule_generator
    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv

    import multi_agent_training
    from reinforcement_learning.ppo_agent import PPOPolicy
    from reinforcement_learning.deadlockavoidance_with_decision_agent import (
        DeadLockAvoidanceWithDecisionAgent,
    )
    from utils.agent_action_config import get_action_size

    # training params
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
        "hidden_size": 128,
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

    # set up environment
    env_params = {
        # Test_2
        "n_agents": 5,
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

    train_params = Namespace(**train_params)
    env_params = Namespace(**env_params)
    obs_params = Namespace(**obs_params)

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

    # set up policy
    checkpoint = "./checkpoints_sutd/210726042625-800.pth"

    state_size = 17 + 17 + 15 + 5 * 9 + 12 * 2

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

    # set up renderer
    renderer = RenderTool(env, gl="PIL")

    # evaluate
    print("evaluating ...")
    env_images, score, completion, step = eval_policy_visual(
        env,
        tree_observation,
        policy,
        train_params,
        obs_params,
        renderer,
    )

    # generate gif
    print("generating gif ...")
    generate_gif("./test.gif", env_images)
