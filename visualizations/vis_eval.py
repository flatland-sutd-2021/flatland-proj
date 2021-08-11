# builtin modules
import sys
import os
from argparse import Namespace
from typing import List, Dict, Tuple

# external modules
import numpy as np
import imageio
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

# fix mac-specific issues
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# resolve internal imports
import vis_utils
from vis_utils import STARTER_PATH
sys.path.append(os.path.normpath(STARTER_PATH))

# internal modules
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent
from utils.agent_action_config import get_action_size
from rail_env_record import RailEnvRecord
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
    using_hint: bool = False
) -> Tuple[RailEnvRecord, float, float, int]:

    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    predictor = ShortestPathPredictorForRailEnv(obs_params.observation_max_path_depth)
    action_dict = dict()
    
    # do not regenerate rail and schedule since we will regenerate them before policy loading
    obs, info = env.reset(regenerate_rail=False, regenerate_schedule=False)

    policy.reset(env)
    record_env = RailEnvRecord(env)

    if using_hint:
        hint_agent = DeadLockAvoidanceAgent(env, get_action_size(), False)
        hint_agent.reset(env)

    agent_obs = [None] * env.get_num_agents()
    score = 0.0

    max_steps = env._max_episode_steps
    final_step = 0

    print("FOR MAX", max_steps, "STEPS")
    print("EVALUATING POLICY:")

    policy.start_episode(train=False)

    # create reward modifier for the purposes of tracking stops
    eval_mod = RewardModifier(env)
    for step in range(max_steps - 1):
        print(f"\r step: {step:04} ", end="")
        policy.start_step(train=False)

        eval_mod.check_staticness(env)

        # Compute expensive stuff ONCE per step
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

                    if using_hint:
                        hint_agent.start_step(False)
                        hint_agent_action = hint_agent.act(agent, obs[agent], -1)
                        
                        hint = [0, 0, 0, 0, 0]
                        hint[hint_agent_action] = 1

                        print("hint_agent_action:", hint_agent_action)
                        print("hint:", hint)
                        
                        state_vector += hint

                    action = policy.act(agent, state_vector, eps=0.0)


            action_dict.update({agent: int(action)})
        policy.end_step(train=False)

        print(action_dict)
        obs, all_rewards, done, info = env.step(map_actions(action_dict))
        record_env.step(map_actions(action_dict))

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

    return record_env, normalized_score, completion, final_step


def eval_policy_visual_heuristic(
    env: RailEnv,
    tree_observation: ObservationBuilder,
    policy: LearningPolicy,
    train_params: Namespace,
    obs_params: Namespace,
) -> Tuple[RailEnvRecord, float, float, int]:

    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    predictor = ShortestPathPredictorForRailEnv(obs_params.observation_max_path_depth)
    action_dict = dict()
    
    # do not regenerate rail and schedule since we will regenerate them before policy loading
    obs, info = env.reset(regenerate_rail=False, regenerate_schedule=False)

    policy.reset(env)
    record_env = RailEnvRecord(env)

    agent_obs = [None] * env.get_num_agents()
    score = 0.0

    max_steps = env._max_episode_steps
    final_step = 0

    print("FOR MAX", max_steps, "STEPS")
    print("EVALUATING POLICY:")

    policy.start_episode(train=False)

    for step in range(max_steps - 1):
        print(f"\r step: {step:04} ", end="")
        policy.start_step(train=False)

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
                    action = policy.act(agent, obs[agent], -1)
                    action_dict.update({agent: int(action)})
    
        policy.end_step(train=False)

        obs, all_rewards, done, info = env.step(map_actions(action_dict))
        record_env.step(map_actions(action_dict))

        # No reward hacking here
        for agent in env.get_agent_handles():
            score += all_rewards[agent]

        final_step = step

        print(action_dict)
        print(done)
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

    return record_env, normalized_score, completion, final_step


def generate_gif(path: str, record_env: RailEnvRecord):
    # env_images[i] is of shape (?, ?, 4), with each value in the range 0-254
    renderer = RenderTool(record_env, gl="PIL")
    env_images = []

    for i in range(record_env.get_record_length()):
        record_env.set_record_step(i)
        renderer.render_env(show_observations=False)
        env_images.append(renderer.get_image())

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
