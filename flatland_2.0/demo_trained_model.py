from main import action_space, hidden_size,in_channels, out_channels, kernel_size, rnn_num_layers, agent_num, num_epochs, gamma, learning_rate, clip_epsilon, c1, c2, obs_into_right_shape
from PPO import FlatlandPPO
import numpy as np
from numpy import inf
import time
from torch import load
import math
from PPO import Memory
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator # Round 1
from flatland.envs.schedule_generators import sparse_schedule_generator # Round 2
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool

ppo = FlatlandPPO(
				  action_space=action_space,
				  hidden_size=hidden_size,
				  in_channels=in_channels, 
				  out_channels=out_channels, 
				  kernel_size=kernel_size, 
				  agent_num=agent_num, 
				  num_epochs=num_epochs,
				  rnn_num_layers=rnn_num_layers, 
				  gamma=gamma, 
				  learning_rate=learning_rate, 
				  clip_epsilon=clip_epsilon, 
				  c1=c1, 
				  c2=c2)


ppo.model.load_state_dict(load('/home/vache/ML_projects/rl/flatland_2.0/saved_model/PPO_flatland.pth'))


episode_num = 10
agent_num=1
env_width=8
env_height=8
rail_generator=complex_rail_generator(
									nr_start_goal=9,
									nr_extra=1,
									min_dist=9,
									max_dist=99999,
									seed=0)
schedule_generator=complex_schedule_generator()
env = RailEnv(
        width=env_width,
        height=env_height,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        obs_builder_object=GlobalObsForRailEnv(),
        number_of_agents=agent_num)

env_renderer = RenderTool(env)
render = True
render_sleep_time = 0.03
stuck_break_pont = 20
max_timesteps_in_episode = 1000

timesteps = 0
episode_rewards = []

for episode in range(episode_num):
	all_done = False
	episode_timesteps = 0
	print('Episode:', episode)
	states_dict = env.reset()
	states = obs_into_right_shape(list(states_dict.values()), env)
	reward_sum = 0

	if render:
		env_renderer.reset()


	while not all_done:

		timesteps += 1
		episode_timesteps += 1

		actions, values = ppo.step(obs=states, model=ppo.model_old, agent_num=env.number_of_agents, greedy=True)

		states, rewards, done, info = env.step(actions)
		malfunction = info['malfunction']
		states = obs_into_right_shape(list(states.values()), env)

		all_done = done['__all__']

		done_sum = 0
		for agent in range(env.number_of_agents):
			if malfunction[agent] == 1:
				done[agent] = True
				print('malfunction', agent)

			done_sum += int(done[agent])

		all_done = (done_sum == env.number_of_agents)

		if episode_timesteps >= max_timesteps_in_episode:
			print('############FOREVEEEER###############')
			break

		if render:
			env_renderer.render_env(show=True, frames=False, show_observations=False)
			time.sleep(render_sleep_time)

		reward_sum += sum(list(rewards.values()))

	print('Episode:{} reward:{}'.format(episode, reward_sum))
