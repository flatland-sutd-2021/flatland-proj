from pathlib import Path
import sys
import os
import random
import math

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from flatland_sutd import *

from argparse import ArgumentParser, Namespace
from collections import deque
from datetime import datetime
from pprint import pprint

import numpy as np
import psutil
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning.dddqn_policy import DDDQNPolicy
from reinforcement_learning.deadlockavoidance_with_decision_agent import DeadLockAvoidanceWithDecisionAgent
from reinforcement_learning.multi_decision_agent import MultiDecisionAgent
from reinforcement_learning.ppo_agent import PPOPolicy
from utils.agent_action_config import get_flatland_full_action_size, get_action_size, map_actions, map_action, \
    set_action_size_reduced, set_action_size_full, map_action_policy
from utils.dead_lock_avoidance_agent import DeadLockAvoidanceAgent

from utils.timer import Timer
from utils.observation_utils import normalize_observation
from utils.fast_tree_obs import FastTreeObs


"""
This file shows how to train multiple agents using a reinforcement learning approach.
After training an agent, you can submit it straight away to the NeurIPS 2020 Flatland challenge!

Agent documentation: https://flatland.aicrowd.com/getting-started/rl/multi-agent.html
Submission documentation: https://flatland.aicrowd.com/getting-started/first-submission.html
"""


def create_rail_env(env_params, tree_observation):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        malfunction_generator=ParamMalfunctionGen(malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=None
    )


def train_agent(train_params, train_env_params, eval_env_params, obs_params):
    if True:
        # CH3: Training plan
        # SET INITIAL PARAMS

        SUCCESS_THRESHOLD = 9999999999
        train_env_params.n_agents = 3
        train_env_params.x_dim = 25
        train_env_params.y_dim = 25
        train_env_params.n_cities = 2
        train_env_params.max_rails_between_cities = 2
        train_env_params.max_rails_in_city = 4

        obs_params.observation_tree_depth = 4

    # Environment parameters
    n_agents = train_env_params.n_agents
    x_dim = train_env_params.x_dim
    y_dim = train_env_params.y_dim
    n_cities = train_env_params.n_cities
    max_rails_between_cities = train_env_params.max_rails_between_cities
    max_rails_in_city = train_env_params.max_rails_in_city
    seed = train_env_params.seed
    number_of_agents = n_agents

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes
    restore_replay_buffer = train_params.restore_replay_buffer
    save_replay_buffer = train_params.save_replay_buffer

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    if not train_params.use_fast_tree_observation:
        print("\nUsing standard TreeObs")

        def check_is_observation_valid(observation):
            return observation

        def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
            return normalize_observation(observation, tree_depth, observation_radius)

        tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)
        tree_observation.check_is_observation_valid = check_is_observation_valid
        tree_observation.get_normalized_observation = get_normalized_observation
    else:
        print("\nUsing FastTreeObs")

        def check_is_observation_valid(observation):
            return True

        def get_normalized_observation(observation, tree_depth: int, observation_radius=0):
            return observation

        tree_observation = FastTreeObs(max_depth=observation_tree_depth)
        tree_observation.check_is_observation_valid = check_is_observation_valid
        tree_observation.get_normalized_observation = get_normalized_observation

    # Setup the environments
    train_env = create_rail_env(train_env_params, tree_observation)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(eval_env_params, tree_observation)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    if not train_params.use_fast_tree_observation:
        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = train_env.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
        state_size = n_features_per_node * n_nodes
    else:
        # Calculate the state size given the depth of the tree observation and the number of features
        state_size = tree_observation.observation_dim

    # ABLATION STUDY: Change state size here
    if True:
        # k_best_paths: 17 + 17
        # root_extra: 15 + k * 9
        # rvnn children: 12 * k_branches
        state_size = (
            17 + 17
            + 15 + 5 * 9
            + 12 * 2
            # + 5
        )

    action_count = [0] * get_flatland_full_action_size()
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [2] * n_agents
    update_values = [False] * n_agents

    # Smoothed values used as target for hyperparameter tuning
    smoothed_eval_normalized_score = -1.0
    smoothed_eval_completion = 0.0

    scores_window = deque(maxlen=checkpoint_interval)  # todo smooth when rendering instead
    rewards_window = deque(maxlen=checkpoint_interval)  # todo smooth when rendering instead
    completion_window = deque(maxlen=checkpoint_interval)

    if train_params.action_size == "reduced":
        set_action_size_reduced()
    else:
        set_action_size_full()

    # Double Dueling DQN policy
    if train_params.policy == "DDDQN":
        policy = DDDQNPolicy(state_size, get_action_size(), train_params)
    elif train_params.policy == "SUTD":
        inter_policy = PPOPolicy(state_size, get_action_size(), use_replay_buffer=False, in_parameters=train_params)
        policy = DeadLockAvoidanceWithDecisionAgent(train_env, state_size, get_action_size(), inter_policy)
    elif train_params.policy == "PPO":
        policy = PPOPolicy(state_size, get_action_size(), use_replay_buffer=False, in_parameters=train_params)
    elif train_params.policy == "DeadLockAvoidance":
        policy = DeadLockAvoidanceAgent(train_env, get_action_size(), enable_eps=False)
    elif train_params.policy == "DeadLockAvoidanceWithDecision":
        # inter_policy = PPOPolicy(state_size, get_action_size(), use_replay_buffer=False, in_parameters=train_params)
        inter_policy = DDDQNPolicy(state_size, get_action_size(), train_params)
        policy = DeadLockAvoidanceWithDecisionAgent(train_env, state_size, get_action_size(), inter_policy)
    elif train_params.policy == "MultiDecision":
        policy = MultiDecisionAgent(state_size, get_action_size(), train_params)
    else:
        policy = PPOPolicy(state_size, get_action_size(), use_replay_buffer=False, in_parameters=train_params)

    # make sure that at least one policy is set
    if policy is None:
        policy = DDDQNPolicy(state_size, get_action_size(), train_params)

    # Load existing policy
    if train_params.load_policy != "":
        policy.load(train_params.load_policy)

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print("\n???? Could't load replay buffer, were the experiences generated using the same tree depth?")
            print(e)
            exit(1)

    print("\n???? Replay buffer status: {}/{} experiences".format(len(policy.memory.memory), train_params.buffer_size))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print(
            "??????  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left.".format(
                hdd.free / (2 ** 30)))

    # TensorBoard writer
    writer = SummaryWriter(comment="_" + train_params.policy + "_" + train_params.action_size)

    training_timer = Timer()
    training_timer.start()

    print(
        "\n???? Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
            train_env.get_num_agents(),
            x_dim, y_dim,
            n_episodes,
            n_eval_episodes,
            checkpoint_interval,
            training_id
        ))

    success_count = 0

    for episode_idx in range(n_episodes + 1):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        if train_params.n_agent_fixed:
            number_of_agents = n_agents
            train_env_params.n_agents = n_agents
        else:
            if True: # CH3: Incrementing difficulty
                if (episode_idx % (n_episodes // 30)) == 1 or success_count > SUCCESS_THRESHOLD:
                    success_count = 0

                    # Make an eval env based on the PREVIOUS difficulty!!
                    eval_env = create_rail_env(train_env_params, tree_observation)
                    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

                    print(f"\n\n== INCREMENTING DIFFICULTY (Episode: {episode_idx}) ==")
                    train_env_params.n_agents += math.ceil(10**(len(str(train_env_params.n_agents))-1)*0.75)
                    train_env_params.n_cities = train_env_params.n_agents // 10 + 2

                    train_env_params.x_dim = math.ceil(math.sqrt((2*(math.ceil(train_env_params.max_rails_in_city/2) + 3)) ** 2 * (1.5*train_env_params.n_cities)))+7
                    train_env_params.y_dim = train_env_params.x_dim

                    # Malfunction interval is the minimum number of intervals between malfunctions
                    # As agent count increases, malfunctions from individual agents become MORE SPARSE
                    # since more time is allotted to completing the environmenet
                    malfunction_interval = int(250 * (episode_idx // (n_episodes // 11)))
                    if malfunction_interval == 0:
                        train_env_params.malfunction_rate = 0
                    else:
                        train_env_params.malfunction_rate = 1 / malfunction_interval

                    # Environment parameters
                    n_agents = train_env_params.n_agents
                    x_dim = train_env_params.x_dim
                    y_dim = train_env_params.y_dim
                    n_cities = train_env_params.n_cities
                    max_rails_between_cities = train_env_params.max_rails_between_cities
                    max_rails_in_city = train_env_params.max_rails_in_city
                    seed = train_env_params.seed
                    number_of_agents = n_agents

                    print(f"n_agents: {n_agents} | map_size: ({x_dim},{y_dim}) | malfunction_rate: {train_env_params.malfunction_rate}\n\n")

                    agent_obs = [None] * n_agents
                    agent_prev_obs = [None] * n_agents
                    agent_prev_action = [2] * n_agents
                    update_values = [False] * n_agents
                    policy.report_selector()

            else:
                number_of_agents = int(min(n_agents, 1 + np.floor(episode_idx / 200)))
                train_env_params.n_agents = episode_idx % number_of_agents + 1

        train_env = create_rail_env(train_env_params, tree_observation)
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
        policy.reset(train_env)
        reset_timer.end()

        if train_params.render:
            # Setup renderer
            env_renderer = RenderTool(train_env, gl="PGL")
            env_renderer.set_new_rail()

        score = 0
        reward_tracker = 0

        nb_steps = 0
        actions_taken = []

        # Compute expensive stuff ONCE per step
        if True: # CH3: Set to True when ready
            agent_positions, agent_handles = get_agent_positions(train_env)
            kd_tree = KDTree(agent_positions)

            # This is -NOT- total agent count or active agent count!
            num_agents_on_map = get_num_agents_on_map(train_env)
            hint_agent = DeadLockAvoidanceAgent(train_env, get_action_size(), False)

        # Build initial agent-specific observations
        for agent_handle in train_env.get_agent_handles():
            if tree_observation.check_is_observation_valid(obs[agent_handle]):
                if True: # CH3: When it is time...
                    # NOTE: This bit might look unecessary, but it's actually
                    # needed to populate agent_prev_obs...
                    # rvnn_out = policy.rvnn(obs[agent_handle])
                    hint = [0, 0, 0, 0, 0]
                    # hint[hint_agent.act(agent_handle, obs[agent_handle], -1)] = 1
                    state_vector = [
                        # == ROOT ==
                        *get_k_best_node_states(obs[agent_handle], train_env, num_agents_on_map, obs_params.observation_tree_depth),

                        # == ROOT EXTRA ==
                        train_env.number_of_agents,
                        *get_self_extra_states(train_env, obs, agent_handle),
                        get_agent_priority_naive(train_env, predictor)[agent_handle],
                        0, # staticness
                        *get_self_extra_knn_states(train_env, agent_handle, agent_handles, kd_tree, k_num=5),

                        # == RVNN CHILDREN ==
                        *rvnn_out,
                        # *hint
                    ]


                    # print("STATE VECTOR SIZE:", len(state_vector))
                    # print(state_vector)

                    agent_obs[agent_handle] = state_vector # CH3: OBS HACK IS HERE!!!
                else:
                    agent_obs[agent_handle] = tree_observation.get_normalized_observation(obs[agent_handle],
                                                                                          observation_tree_depth,
                                                                                          observation_radius=observation_radius)

                agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()

        # Max number of steps per episode
        # This is the official formula used during evaluations
        # See details in flatland.envs.schedule_generfators.sparse_schedule_generator
        # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
        max_steps = train_env._max_episode_steps

        # Run episode
        policy.start_episode(train=True)

        # hacked rewards object starts here
        reward_mod = RewardModifier(train_env)
        valid_action_penalties = {}

        for step in range(max_steps - 1):
            inference_timer.start()
            policy.start_step(train=True)

            # Compute expensive stuff ONCE per step
            if True: # CH3: Set to True when ready
                agent_positions, agent_handles = get_agent_positions(train_env)
                kd_tree = KDTree(agent_positions)

                # This is -NOT- total agent count or active agent count!
                num_agents_on_map = get_num_agents_on_map(train_env)

            for agent_handle in train_env.get_agent_handles():
                agent = train_env.agents[agent_handle]
                if info['action_required'][agent_handle]:
                    if True: # CH3: When it is time...
                        # ABLATION STUDY: Remove RVNN
                        # rvnn_out = policy.rvnn(obs[agent_handle])
                        hint = [0, 0, 0, 0, 0]
                        # hint[hint_agent.act(agent_handle, obs[agent_handle], -1)] = 1
                        state_vector = [
                            # == ROOT ==
                            *get_k_best_node_states(obs[agent_handle], train_env, num_agents_on_map, obs_params.observation_tree_depth),

                            # == ROOT EXTRA ==
                            train_env.number_of_agents,
                            *get_self_extra_states(train_env, obs, agent_handle),
                            get_agent_priority_naive(train_env, predictor)[agent_handle],
                            reward_mod.stop_dict[agent_handle], # staticness
                            *get_self_extra_knn_states(train_env, agent_handle, agent_handles, kd_tree, k_num=5),

                            # == RVNN CHILDREN ==
                            *rvnn_out,
                            # *hint
                        ]

                        agent_obs[agent_handle] = state_vector # CH3: OBS HACK IS HERE!!!

                    update_values[agent_handle] = True
                    action = policy.act(agent_handle, agent_obs[agent_handle], eps=eps_start)

                    if True:
                        if action not in get_valid_actions(train_env, agent_handle):
                            valid_action_penalties[agent_handle] = -5
                        else:
                            valid_action_penalties[agent_handle] = 0

                        # valid_action_penalties[agent_handle] = 0

                    action_count[map_action(action)] += 1
                    actions_taken.append(map_action(action))
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent_handle] = False
                    action = 0
                action_dict.update({agent_handle: action})

            policy.end_step(train=True)
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(map_actions(action_dict))

            # CH3: HACK REWARDS HERE
            if True:
                # Check one step before final step because weird things happen on the last step
                if (step == max_steps - 2):
                    final = True
                else:
                    final = False
                modded_rewards = reward_mod.check_rewards(train_env, all_rewards, final)

            step_timer.end()

            # Render an episode at some interval
            if train_params.render:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent_handle in train_env.get_agent_handles():
                if update_values[agent_handle] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    policy.step(agent_handle,
                                agent_prev_obs[agent_handle],
                                map_action_policy(agent_prev_action[agent_handle]),
                                modded_rewards[agent_handle] + valid_action_penalties[agent_handle], # CH3: HACK REWARDS HERE, DON'T USE ALL_REWARDS
                                agent_obs[agent_handle],
                                done[agent_handle])
                    learn_timer.end()

                    agent_prev_obs[agent_handle] = agent_obs[agent_handle].copy()
                    agent_prev_action[agent_handle] = action_dict[agent_handle]

                # Preprocess the new observations
                if tree_observation.check_is_observation_valid(next_obs[agent_handle]):
                    preproc_timer.start()

                    if True: # CH3: When it's time...
                        pass # Yep, because we do this step ontop
                    else:
                        agent_obs[agent_handle] = tree_observation.get_normalized_observation(next_obs[agent_handle],
                                                                                              observation_tree_depth,
                                                                                              observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent_handle] # For evaluation only
                reward_tracker += modded_rewards[agent_handle]

            nb_steps = step

            if done['__all__']:
                break

        policy.end_episode(train=True)
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())

        if completion == 1:
            success_count += 1
        else:
            success_count = 0

        normalized_score = score / (max_steps * train_env.get_num_agents())
        scores_window.append(normalized_score)
        smoothed_normalized_score = np.mean(scores_window)

        normalized_rewards = reward_tracker / (max_steps * train_env.get_num_agents())
        rewards_window.append(normalized_rewards)
        smoothed_normalized_rewards = np.mean(rewards_window)

        action_probs = action_count / max(1, np.sum(action_count))
        completion_window.append(completion)
        smoothed_completion = np.mean(completion_window)

        if train_params.render:
            env_renderer.close_window()

        # Print logs
        if episode_idx % checkpoint_interval == 0 and episode_idx > 0:
            if not os.path.isdir("./checkpoints"):
                print("MAKING CHECKPOINTS DIRECTORY")
                os.mkdir("./checkpoints")

            policy.save('./checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')

            if save_replay_buffer:
                policy.save_replay_buffer('./replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            # reset action count
            action_count = [0] * get_flatland_full_action_size()

        print(
            '\r???? Episode {}'
            ' | ???? nAgents {:2}/{:2}'
            ' ???? Score: {:7.3f}'
            ' Avg: {:7.3f}'
            ' Train Rewards: {:7.3f}'
            ' Avg: {:7.3f}'
            ' | ???? Done: {:6.2f}%'
            ' Avg: {:6.2f}%'
            # ' | ???? Epsilon: {:.3f} '
            ' | ???? Action Probs: {}'.format(
                episode_idx,
                train_env_params.n_agents, number_of_agents,
                normalized_score,
                smoothed_normalized_score,
                normalized_rewards,
                smoothed_normalized_rewards,
                100 * completion,
                100 * smoothed_completion,
                # eps_start,
                format_action_prob(action_probs)
            ), end=" ", flush=True)

        # Evaluate policy and log results at some interval
        # Skip the first evaluation window
        if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0 and episode_idx > 10:
            scores, completions, nb_steps_eval = eval_policy(eval_env,
                                                             tree_observation,
                                                             policy,
                                                             train_params,
                                                             obs_params)

            writer.add_scalar("evaluation/scores_min", np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max", np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean", np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std", np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores", np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min", np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max", np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean", np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std", np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions", np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min", np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max", np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean", np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std", np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps", np.array(nb_steps_eval), episode_idx)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (
                    1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score", smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion", smoothed_eval_completion, episode_idx)

        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score", smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/rewards", normalized_rewards, episode_idx)
        writer.add_scalar("training/smoothed_rewards", smoothed_normalized_rewards, episode_idx)
        writer.add_scalar("training/completion", np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion", np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_scalar("training/n_agents", train_env_params.n_agents, episode_idx)
        writer.add_histogram("actions/distribution", np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar("actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar("actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar("actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar("actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/epsilon", eps_start, episode_idx)
        writer.add_scalar("training/buffer_size", len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar("timer/total", training_timer.get_current(), episode_idx)
        writer.add_scalar("training/selector_proportion", policy.get_selector_proportion(), episode_idx)
        writer.flush()


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["???", "???", "???", "???", "???"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, tree_observation, policy, train_params, obs_params):
    print("", flush=True)
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius

    predictor = ShortestPathPredictorForRailEnv(obs_params.observation_max_path_depth)

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        print(f"\rEVALUATING POLICY: {episode_idx}/{n_eval_episodes}", end="", flush=True)

        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        policy.reset(env)
        final_step = 0

        policy.start_episode(train=False)

        # create reward modifier for the purposes of tracking stops
        eval_mod = RewardModifier(env)
        for step in range(max_steps - 1):
            policy.start_step(train=False)

            eval_mod.check_staticness(env)

            # Compute expensive stuff ONCE per step
            if True: # CH3: Set to True when ready
                agent_positions, agent_handles = get_agent_positions(env)
                kd_tree = KDTree(agent_positions)

                # This is -NOT- total agent count or active agent count!
                num_agents_on_map = get_num_agents_on_map(env)
                hint_agent = DeadLockAvoidanceAgent(env, get_action_size(), False)

            for agent in env.get_agent_handles():
                if tree_observation.check_is_observation_valid(agent_obs[agent]):
                    agent_obs[agent] = tree_observation.get_normalized_observation(obs[agent], tree_depth=tree_depth,
                                                                                   observation_radius=observation_radius)

                action = 0
                if info['action_required'][agent]:
                    if True or tree_observation.check_is_observation_valid(agent_obs[agent]):
                        if True: # CH3: When it is time...
                            # rvnn_out = policy.rvnn(obs[agent])
                            hint = [0, 0, 0, 0, 0]
                            # hint[hint_agent.act(agent, obs[agent], -1)] = 1
                            state_vector = [
                                # == ROOT ==
                                *get_k_best_node_states(obs[agent], env, num_agents_on_map, obs_params.observation_tree_depth),

                                # == ROOT EXTRA ==
                                env.number_of_agents,
                                *get_self_extra_states(env, obs, agent),
                                get_agent_priority_naive(env, predictor)[agent],
                                eval_mod.stop_dict[agent], # staticness
                                *get_self_extra_knn_states(env, agent, agent_handles, kd_tree, k_num=5),

                                # == RVNN CHILDREN ==
                                *rvnn_out,
                                # *hint
                            ]

                            action = policy.act(agent, state_vector, eps=0.0)
                        else:
                            agent_obs[agent] = tree_observation.get_normalized_observation(obs[agent],
                                                                                           tree_depth,
                                                                                           observation_radius=observation_radius)
                            action = policy.act(agent, agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})
            policy.end_step(train=False)
            obs, all_rewards, done, info = env.step(map_actions(action_dict))

            # No reward hacking here

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        policy.end_episode(train=False)
        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\n\n ??? Eval: score {:.3f} done {:.1f}%\n".format(np.mean(scores), np.mean(completions) * 100.0), flush=True)
    policy.report_selector()

    return scores, completions, nb_steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=5000, type=int)
    parser.add_argument("--n_agent_fixed", help="hold the number of agent fixed", action='store_true')
    parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=1,
                        type=int)
    parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=1,
                        type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=5, type=int)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
    parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
    parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
    parser.add_argument("--eps_decay", help="exploration decay", default=0.9975, type=float)
    parser.add_argument("--buffer_size", help="replay buffer size", default=int(32_000), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="", type=str)
    parser.add_argument("--save_replay_buffer", help="save replay buffer at each evaluation interval", default=False,
                        type=bool)
    parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.97, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=0.5e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
    parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=256, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=10, type=int)
    parser.add_argument("--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=8, type=int)
    parser.add_argument("--render", help="render 1 episode in 100", action='store_true')
    parser.add_argument("--load_policy", help="policy filename (reference) to load", default="", type=str)
    parser.add_argument("--use_fast_tree_observation", help="use FastTreeObs instead of stock TreeObs",
                        action='store_true')
    parser.add_argument("--max_depth", help="max depth", default=2, type=int)
    parser.add_argument("--policy",
                        help="policy name [DDDQN, PPO, DeadLockAvoidance, DeadLockAvoidanceWithDecision, MultiDecision]",
                        default="DeadLockAvoidance")
    parser.add_argument("--action_size", help="define the action size [reduced,full]", default="full", type=str)

    training_params = parser.parse_args()
    env_params = [
        {
            # Test_0
            "n_agents": 1,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 5,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 100,
            "seed": 0
        },
        {
            # Test_3
            "n_agents": 20,
            "x_dim": 35,
            "y_dim": 35,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
        {
            # Test_3
            "n_agents": 58,
            "x_dim": 40,
            "y_dim": 40,
            "n_cities": 5,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
    ]

    obs_params = {
        "observation_tree_depth": training_params.max_depth,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }


    def check_env_config(id):
        if id >= len(env_params) or id < 0:
            print("\n???? Invalid environment configuration, only Test_0 to Test_{} are supported.".format(
                len(env_params) - 1))
            exit(1)


    check_env_config(training_params.training_env_config)
    check_env_config(training_params.evaluation_env_config)

    training_env_params = env_params[training_params.training_env_config]
    evaluation_env_params = env_params[training_params.evaluation_env_config]

    # FIXME hard-coded for sweep search
    # see https://wb-forum.slack.com/archives/CL4V2QE59/p1602931982236600 to implement properly
    # training_params.use_fast_tree_observation = True

    print("\nTraining parameters:")
    pprint(vars(training_params))
    print("\nTraining environment parameters (Test_{}):".format(training_params.training_env_config))
    pprint(training_env_params)
    print("\nEvaluation environment parameters (Test_{}):".format(training_params.evaluation_env_config))
    pprint(evaluation_env_params)
    print("\nObservation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
    train_agent(training_params, Namespace(**training_env_params), Namespace(**evaluation_env_params),
                Namespace(**obs_params))
