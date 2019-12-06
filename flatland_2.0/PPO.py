from torch.distributions import Categorical
from torch import device, cuda, from_numpy, clamp, exp, min, squeeze, mean, optim, tensor, FloatTensor, stack, cat, sum
import torch.nn as nn
from flatland_model import CNN_RNN
import numpy as np
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

device = device("cuda:0" if cuda.is_available() else "cpu")


class Memory:
	def __init__(self):
		self.actions = {}
		self.states = {}
		self.logprobs = {}
		self.rewards = {}
		self.values = {}

	def clear_memory(self):
		self.actions.clear()
		self.states.clear()
		self.logprobs.clear()
		self.rewards.clear()
		self.values.clear()

	def flat_mem_values(self, agent_num):
		"""
		As values are appended by lists, so we need to flatten it before using
		"""
		for agent in range(agent_num):
			self.values[agent] = np.fromiter(chain.from_iterable(self.values[agent]), dtype='float')
			self.values[agent] = tensor(self.values[agent])


	def init_vals(self, agent_num):
		for agent in range(agent_num):
			self.actions[agent] = []
			self.logprobs[agent] = []
			self.values[agent] = []
			self.states[agent] = []
			self.rewards[agent] = []

	def init_rewards(self, agent_num):
		for agent in range(agent_num):
			self.rewards[agent] = []



class FlatlandPPO():

	def __init__(self, action_space, hidden_size, in_channels, out_channels, kernel_size, agent_num, num_epochs, rnn_num_layers, gamma=0.95, learning_rate=0.001, clip_epsilon=0.2, c1=0.5, c2=0.01, betas=(0.9, 0.999)):

		self.model = CNN_RNN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, hidden_size=hidden_size, bidirectional=True, rnn_num_layers=rnn_num_layers, action_size=action_space)
		self.model_old = CNN_RNN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, hidden_size=hidden_size, bidirectional=True, rnn_num_layers=rnn_num_layers, action_size=action_space)
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.clip_epsilon = clip_epsilon
		self.num_epochs = num_epochs
		self.c1 = c1
		self.c2 = c2
		self.betas = betas
		self.agent_num = agent_num
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas)
		self.MseLoss = nn.MSELoss()

		self.step_counter = 0
		self.prev_value = 0
		self.curr_value = 0
		self.prev_policy = 0
		self.curr_policy = 0



	def evaluate(self, state, action, model):

		"""
		Based on the policy (model) returns that model's policy information,
		log probs for actions, state values and policy action distribution entropy
		"""

		value, action_probs = model.forward(state)
		dist = Categorical(action_probs)

		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()


		return action_logprobs, squeeze(value), dist_entropy


	def step(self, obs, model, agent_num, memory=None, greedy=False):

		"""
		Does a step and also appends needed data to memory
		obs is a numpy array
		"""
		agent_actions = {}
		torch_obs = tensor(obs).float().to(device)
		# torch_obs = torch_obs.view(1, agent_num, torch_obs.size(0), torch_obs.size(1), torch_obs.size(2))
		values, action_probs = model.forward(torch_obs)

		dist = Categorical(action_probs)

		if greedy:
			actions = action_probs.max(-1)[1]

		else:
			actions = dist.sample()

		actions_for_dict = squeeze(actions)

		
		for agent in range(agent_num):
			if agent_num == 1:
				agent_actions[agent] = int(actions_for_dict)
			else:
				agent_actions[agent] = int(actions_for_dict[agent].item())

			if memory:
				memory.states[agent].append(torch_obs[0][agent])
				memory.actions[agent].append(actions[agent])
				memory.logprobs[agent].append(dist.log_prob(actions[agent])[agent])

		return agent_actions, values

	def calculate_td_lambda(self, memory):
		"""
		Calculates td lambda values and stores it in memory

		input: memory object
		output: saves in memory.values - a dict of list, a list of values for every agent
		"""
		gamma_power = [i for i in range(len(memory.rewards[0]))]
		powerful_gamma = np.power(self.gamma, gamma_power)
		weighted_average = np.multiply((1-self.gamma), powerful_gamma)

		for agent in range(len(memory.rewards)):
			if memory.rewards[agent]:
				agent_values = [np.sum(memory.rewards[agent][i:]) for i in range(len(memory.rewards[agent]))]
				agent_values = np.multiply(weighted_average, agent_values)
				memory.values[agent].append(list(agent_values))
				memory.rewards[agent] = []



	def train(self, memory, agent_num, width, height):

		print('---------TRAINING---------')

		torch_states = list(chain.from_iterable(list(memory.states.values())))
		torch_actions = list(chain.from_iterable(list(memory.actions.values())))
		torch_logprobs = list(chain.from_iterable(list(memory.logprobs.values())))
		torch_values = list(chain.from_iterable(list(memory.values.values())))

		torch_states = stack(torch_states).view(-1, agent_num, 22, width, height).detach().to(device)
		torch_actions = stack(torch_actions).view(agent_num, -1).detach().to(device)
		torch_logprobs = stack(torch_logprobs).view(agent_num, -1).detach().to(device)
		torch_values = stack(torch_values).view(agent_num, -1).detach().to(device)
		# Normalizing the values:
		# torch_values = (torch_values - torch_values.mean()) / (torch_values.std() + 1e-5)

		for _ in range(self.num_epochs):

			logprobs, pred_values, dist_entropy = self.evaluate(torch_states, torch_actions, self.model)
			# print('OLD', torch_values, 'PRED', pred_values)
			advantages = torch_values - pred_values.detach()
			#, 'ACTI', torch_actions)

			# Calculate ratios
			ratios = exp(logprobs - torch_logprobs.detach())
			# print('RATIOS', ratios, 'OLD', torch_logprobs, 'PRED', logprobs)


			# The Loss
			part1 = ratios * advantages

			part2 = clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages


			stacked_part1_part2 = stack([part1, part2])

			loss = -min(stacked_part1_part2, 0)[0] + self.c1*self.MseLoss(pred_values.float(), torch_values.float()).view(-1, 1) - self.c2*dist_entropy

			

			# take gradient step
			self.optimizer.zero_grad()
			loss_mean = loss.mean()

			loss_mean.backward()

			# for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
			# 	print(p.grad.data.norm(2).item())


			self.optimizer.step()


		# update old model's weights with the current one

		self.model_old.load_state_dict(self.model.state_dict())

		
        


