from .state_construction import *

class RewardModifier:
	def __init__ (self, train_env):
		agent_handles = train_env.get_agent_handles()
		self.reward_dict = dict.fromkeys(agent_handles, 0)
		self.agent_list = agent_handles
		self.stop_dict = dict.fromkeys(agent_handles, 0) # counts the consecutive stops
		self.prev_position = self.get_agent_pos(train_env)
		self.prev_distance = self.get_all_agent_dist(train_env)

	def reset_rewards(self):
		self.reward_dict = dict.fromkeys(self.reward_dict, 0)

	def get_agent_pos(self, train_env):
		agent_pos, agent_handles = get_agent_positions(train_env)

		pos_dict = {}
		for i in range(len(agent_pos)):
			pos_dict[agent_handles[i]] = agent_pos[i]

		return pos_dict

	def get_all_agent_dist(self, train_env):
		dist_dict = {}
		for handle in self.agent_list:
			dist_dict[handle] = self.get_agent_dist(train_env, handle)

		return dist_dict

	def get_agent_dist(self, train_env, handle):
		# code to get agent distance from target
		distance_map = train_env.distance_map.get()
		agent = train_env.agents[handle]

		position, direction = agent.position, agent.direction

		if position is None:
			position = agent.initial_position

		if direction is None:
			direction = agent.initial_direction

		return distance_map[handle, position[0], position[1], direction]

	def check_staticness(self, train_env):
		cur_pos = self.get_agent_pos(train_env)

		for handle in self.agent_list:
			agent_status = train_env.agents[handle].status

			if (agent_status != RailAgentStatus.DONE_REMOVED) and (agent_status != RailAgentStatus.DONE) and (train_env.agents[handle].malfunction_data['malfunction'] == 0):
				if (cur_pos[handle] == self.prev_position[handle]):
					self.stop_dict[handle] += 1

				# Penalise stopping
				if (self.stop_dict[handle] > 0) and (self.stop_dict[handle] < 10):
					self.reward_dict[handle] -= 1

				# Heavily penalise deadlock
				elif (self.stop_dict[handle] > 10):
					self.reward_dict[handle] -= 10
			else:
				self.stop_dict[handle] == 0
				self.reward_dict[handle] += 10

		self.prev_position = cur_pos

	def print_rewards(self, modded_dict):
		for handle in self.agent_list:
			reward = modded_dict[handle]
			print(str(handle) + ": " +str(reward))

	# Check if agent is getting closer or not
	def check_step(self, train_env):
		for handle in self.agent_list:
			agent_status = train_env.agents[handle].status

			if (agent_status != RailAgentStatus.DONE_REMOVED) and (agent_status != RailAgentStatus.DONE) and (train_env.agents[handle].malfunction_data['malfunction'] == 0):
				dist = self.get_agent_dist(train_env, handle)

				if dist < self.prev_distance[handle]:
					self.reward_dict[handle] -= 0.35
				elif dist > self.prev_distance[handle]:
					self.reward_dict[handle] -= 0.5

			# Reward reaching target sooner
			if agent_status == RailAgentStatus.DONE_REMOVED:
				self.reward_dict[handle] += 5

		self.prev_distance = self.get_all_agent_dist(train_env)

	def check_rewards(self, train_env, all_reward, final_check=False):
		self.reset_rewards()
		self.check_staticness(train_env)
		self.check_step(train_env)

		if final_check:
			self.final_check(train_env)

		return self.modify_rewards(all_reward)

	def modify_rewards(self, all_reward):
		modified_rewards = dict.fromkeys(all_reward, 0)
		for handle in self.agent_list:
			modified_rewards[handle] = all_reward[handle] + self.reward_dict[handle]

		return modified_rewards

	def final_check(self, train_env):
		agent_state, agent_handles = get_agent_state

		for handle in self.agent_list:
			agent_status = train_env.agents[handle]

			if (agent_status != RailAgentStatus.DONE_REMOVED) and (agent_status != RailAgentStatus.DONE):
				self.reward_dict[handle] -= 250
