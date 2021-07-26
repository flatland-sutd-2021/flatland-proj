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


def eval_policy_visual(
    env: RailEnv,
    tree_observation: ObservationBuilder,
    policy: LearningPolicy,
    obs_params: Dict,
    renderer: RenderTool,
) -> Tuple[List[np.ndarray], float, float, int]:

    # set up environment
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    policy.reset(env)

    # initialize observations and score
    agent_obs = [None] * env.get_num_agents()
    score = 0.0

    # render initial environment image
    renderer.render_env()
    env_images = [renderer.get_image()]

    policy.start_episode(train=False)

    for step in range(env._max_episode_steps - 1):
        print(f"\r step: {step:04}", end="")
        policy.start_step(train=False)
        action_dict = {}

        # generate action_dict
        for agent_handle in env.get_agent_handles():
            # default action is 0
            action = 0

            # get agent observations
            if tree_observation.check_is_observation_valid(obs[agent_handle]):
                agent_obs[agent_handle] = tree_observation.get_normalized_observation(
                    obs[agent_handle],
                    tree_depth=obs_params["observation_tree_depth"],
                    observation_radius=obs_params["observation_radius"],
                )

            # update action if required
            if info["action_required"][agent_handle]:
                if tree_observation.check_is_observation_valid(obs[agent_handle]):
                    action = policy.act(agent_handle, agent_obs[agent_handle], eps=0.0)

            # add action to action_dict
            action_dict[agent_handle] = action

        policy.end_step(train=False)

        # step environment and render new environment image
        obs, all_rewards, done, info = env.step(map_actions(action_dict))
        renderer.render_env()
        env_images.append(renderer.get_image())

        for agent_handle in env.get_agent_handles():
            score += all_rewards[agent_handle]

        if done["__all__"]:
            print()
            break

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
    completion = tasks_finished / max(1, env.get_num_agents())

    return env_images, score, completion, step


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
    from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
    from utils.agent_action_config import get_action_size

    # set up environment
    env_params = {
        # Test_2
        "n_agents": 10,
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

    predictor = ShortestPathPredictorForRailEnv(
        obs_params["observation_max_path_depth"]
    )

    tree_observation = TreeObsForRailEnv(
        max_depth=obs_params["observation_max_path_depth"],
        predictor=predictor,
    )

    def check_is_observation_valid(observation):
        return observation != None

    def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
        return normalize_observation(observation, tree_depth, observation_radius)

    tree_observation.check_is_observation_valid = check_is_observation_valid
    tree_observation.get_normalized_observation = get_normalized_observation

    env = multi_agent_training.create_rail_env(
        Namespace(**env_params),
        tree_observation,
    )

    # set up policy
    checkpoint = None

    n_features_per_node = tree_observation.observation_dim
    n_nodes = sum(
        [np.power(4, i) for i in range(obs_params["observation_tree_depth"] + 1)]
    )
    state_size = n_features_per_node * n_nodes

    policy = DeadLockAvoidanceAgent(env, get_action_size(), enable_eps=False)
    policy.load(checkpoint)

    # set up renderer
    renderer = RenderTool(env, gl="PIL")

    # evaluate
    print("evaluating ...")
    env_images, score, completion, step = eval_policy_visual(
        env, tree_observation, policy, obs_params, renderer
    )

    # generate gif
    print("generating gif ...")
    generate_gif("./test.gif", env_images)
