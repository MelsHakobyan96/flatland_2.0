import PPO
import numpy as np
from numpy import inf
import time
import matplotlib.pyplot as plt
import math
from torch import device, cuda, from_numpy, clamp, exp, min, squeeze, mean, optim, tensor, save, cat, stack
from PPO import Memory
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator # Round 1
from flatland.envs.schedule_generators import sparse_schedule_generator # Round 2
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool
import warnings
warnings.filterwarnings('ignore')

device = device("cuda:0" if cuda.is_available() else "cpu")


continue_training=False

##################################
# Hyper parameters
rnn_num_layers=2
action_space=5
hidden_size=256
in_channels=22
out_channels=22
kernel_size=2
agent_num=1
num_epochs=8
epoche_per_episode=30
gamma=0.95
learning_rate=0.002
clip_epsilon=0.2
c1=0.3
c2=0.2
update_timestep=900
random_seed=2
episode_num=1000
scheduled_learning=True
env_update_time=50
agent_num_update_time=2
env_update_decay=1
hardness_lvl=1
##################################

##################################
# Env parameters
env_width=11
env_height=11
rail_generator=complex_rail_generator(
								nr_start_goal=20,
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
render_sleep_time = 0.0
stuck_break_pont = 20
max_timesteps_in_episode = update_timestep
##################################


memory = Memory()
memory.init_vals(env.number_of_agents)
ppo = PPO.FlatlandPPO(
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

if continue_training:
	ppo.model.load_state_dict(load('/home/vache/ML_projects/rl/flatland/saved_model/PPO_flatland.pth'))


def env_gradual_update(input_env, agent=False, hardness_lvl=1):

	agent_num = input_env.number_of_agents
	env_width = input_env.width + 4
	env_height = input_env.height + 4

	map_agent_ratio = int(np.round(((env_width+env_height)/2)/5 - 2))

	if map_agent_ratio > 0:
		agent_num = int(np.round(((env_width+env_height)/2)/5 - 2))
	else:
		agent_num = 1

	if hardness_lvl == 1:

		rail_generator=complex_rail_generator(
										nr_start_goal=20,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator=complex_schedule_generator()
	else:

		rail_generator=sparse_rail_generator(
										nr_start_goal=9,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator= sparse_schedule_generator()

	global env, env_renderer, render


	if render:
		env_renderer.close_window()

	env = RailEnv(
	    width=env_width,
	    height=env_height,
	    rail_generator=rail_generator,
	    schedule_generator=schedule_generator,
	    obs_builder_object=GlobalObsForRailEnv(),
	    number_of_agents=agent_num)

	env_renderer = RenderTool(env)


def env_random_update(input_env, decay, agent=False, hardness_lvl=1):

	agent_num = np.random.randint(1, 5)
	env_width = (agent_num + 2) * 5
	env_height = (agent_num + 2) * 5


	if hardness_lvl == 1:

		rail_generator=complex_rail_generator(
										nr_start_goal=20,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator=complex_schedule_generator()
	else:

		rail_generator=sparse_rail_generator(
										nr_start_goal=9,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator= sparse_schedule_generator()

	global env, env_renderer, render


	if render:
		env_renderer.close_window()

	env = RailEnv(
	    width=env_width,
	    height=env_height,
	    rail_generator=rail_generator,
	    schedule_generator=schedule_generator,
	    obs_builder_object=GlobalObsForRailEnv(),
	    number_of_agents=agent_num)

	env_renderer = RenderTool(env)


def obs_into_right_shape(obs, env):

	obs_lst = []
	for agent in range(env.number_of_agents):
		new_obs1 = tensor(obs[agent][0])
		new_obs2 = tensor(obs[agent][1])
		new_obs3 = tensor(obs[agent][2])


		new_obs1 = new_obs1.view(-1, env.height, env.width)
		new_obs2 = new_obs2.view(-1, env.height, env.width)
		new_obs3 = new_obs3.view(-1, env.height, env.width)


		stacked = cat([new_obs1, new_obs2, new_obs3])
		obs_lst.append(stacked)

	stackedd = stack(obs_lst)
	stackedd = stackedd.view(-1, env.number_of_agents, 22, env.height, env.width)

	return stackedd




def main():

	timesteps = 0
	episode_rewards = []
	memory.init_vals(env.number_of_agents)

	for episode in range(episode_num):
		
		
		print('Episode:', episode)
		states_dict = env.reset()
		
		states = obs_into_right_shape(list(states_dict.values()), env)
		reboot_state = states
		

		if render:
			env_renderer.reset()

		for episode_epoch in range(epoche_per_episode):
			print('Episode Epoch', episode_epoch)
			all_done = False
			episode_timesteps = 0
			reward_sum = 0

			while not all_done:

				timesteps += 1
				episode_timesteps += 1

				actions, values = ppo.step(obs=states, model=ppo.model_old, agent_num=env.number_of_agents, memory=memory)

				states, rewards, done, info = env.step(actions)

				malfunction = info['malfunction']
				states = obs_into_right_shape(list(states.values()), env)

				all_done = done['__all__']

				done_sum = 0
				for agent in range(env.number_of_agents):
					memory.rewards[agent].append(rewards[agent])
					if malfunction[agent] == 1:
						done[agent] = True
						rewards[agent] = -100
						print('malfunction', agent)

					done_sum += int(done[agent])

				all_done = (done_sum == env.number_of_agents)

				if episode_timesteps >= max_timesteps_in_episode:
					timesteps = timesteps - episode_timesteps
					memory.init_rewards(env.number_of_agents)
					for agent in range(env.number_of_agents):
						del memory.actions[agent][len(memory.actions[agent])-episode_timesteps:]
						del memory.states[agent][len(memory.states[agent])-episode_timesteps:]
						del memory.logprobs[agent][len(memory.logprobs[agent])-episode_timesteps:]
					episode_timesteps = 0
					print('############FOREVEEEER###############')
					break

				if render:
					env_renderer.render_env(show=True, frames=False, show_observations=False)
					time.sleep(render_sleep_time)

				reward_sum += sum(list(rewards.values()))

			env.restart_agents()
			env.dones = dict.fromkeys(list(range(env.get_num_agents())) + ["__all__"], False)
			states = reboot_state
			print('Reward sum:', reward_sum, 'Agent num', env.number_of_agents, 'Map size {}X{}'.format(env.height, env.width) )

			# print('states', len(memory.states), 'actions', len(memory.actions), 'logprobs', len(memory.logprobs), 'values', len(memory.values))
			episode_rewards.append(reward_sum)
			ppo.calculate_td_lambda(memory)
			if timesteps - update_timestep >= 0:
				timesteps = 0
				memory.flat_mem_values(env.number_of_agents)
				ppo.train(memory=memory, agent_num=env.number_of_agents, width=env.width, height=env.height)
				# if episode % env_update_time == 0:
				# 	env_gradual_update(env)

				memory.init_vals(env.number_of_agents)
				save(ppo.model.state_dict(), '/home/vache/ML_projects/rl/flatland_2.0/saved_model/PPO_flatland.pth')




	plt.plot(episode_rewards)
	plt.ylabel('reward')
	plt.show()




if __name__ == '__main__':
    main()










((env_width+env_height)/2)/5 - 2