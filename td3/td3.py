# Imports 
from torch import nn, tanh
from torch import cat

"""
    ACTOR Network
"""
class ActorNetwork(nn.Module):
    '''
    Actor network for Advantage Actor Critic (TD3) implementation
    '''
    def __init__(self, state_shape, action_count):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )
    
    def forward(self, X):
        x = self.model(X)
        x = tanh(x)
        return x

"""
    Critic Network
"""
class CriticNetwork(nn.Module):
    '''
    Critic network for Advantage Actor Critic (TD3) implementation
    '''
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape + action_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        x = cat([state, action], dim = 1)
        return self.model(x)

from utilities.replay_buffer import ReplayBuffer
import torch.nn.functional as f
from torch import Tensor
import torch
import numpy as np
import os
import pickle
import copy
import matplotlib.pyplot as plt

"""
    TD3 Implementation
"""
class td3:
    def __init__(self, env, params=None):
        self.env = env
        state, info = self.env.reset()
        if(params == None):
            # If no params are provided, use these as defaults
            self.params = {
                'gamma': 0.99,              # discount factor
                'alpha_critic': 0.0001,     # learning rate for critic net
                'alpha_actor': 0.001,      # learning rate for actor network
                'buffer_size': 100000,
                'batch_size' : 64,
                'tau':         0.001,
                'update_rate': 100,
                'noise_scale' : 0.1,
                "actor_update_frequency" : 2,
            }
        else:
            self.params = params

        self.action_count = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high[0]
        self.min_action = self.env.action_space.low[0]
        print(f"Action Count: {self.action_count}, Max Action: {self.max_action}, Min Action: {self.min_action}")
        self.state_size = len(torch.from_numpy(state).float().ravel())
        print(f"State Shape: {self.state_size}")
        self.iterator = 0
        self.actor_update_frequency = self.params["actor_update_frequency"]

        """TD3 Initilizations"""
        # Replay Buffer
        self.buffer = ReplayBuffer(self.params["buffer_size"])
        # Critic Networks
        self.critic_q_network_1 = CriticNetwork(self.state_size, self.action_count)
        self.critic_q_network_2 = CriticNetwork(self.state_size, self.action_count)
        # Actor Network
        self.actor_network = ActorNetwork(self.state_size, self.action_count)
        # Make Critic Network Copies
        self.target_critic_q_network_1 = copy.deepcopy(self.critic_q_network_1)
        self.target_critic_q_network_2 = copy.deepcopy(self.critic_q_network_2)
        # Make Actor Network Copy
        self.target_actor = copy.deepcopy(self.actor_network)
        # Make Optimizers
        self.critic_1_optimizer = torch.optim.Adam(list(self.critic_q_network_1.parameters()) + list(self.critic_q_network_2.parameters()), lr=self.params['alpha_critic'])
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.params['alpha_actor'])


        self.gamma = self.params["gamma"]
        self.tau = self.params["tau"]
        self.batch_size = self.params["batch_size"]
        self.network_update_rate = self.params["update_rate"]
        


        # Initialize Noise
        self.noise_scale = self.params['noise_scale'] # Adjust the noise scale as needed

        # Lists for plotting
        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []
        self.test_rewards = []
        self.noise_arr = []
    
    def change_environment(self, env):
        self.env = env

    def add_noise(self, action):
        # Add Gaussian Noise
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=action.shape)
        self.noise_arr.append(noise)
        return np.clip(action + noise, self.min_action, self.max_action)
    
    def get_action(self, state):
        '''
        Function to get a single action using the actor (policy) network
         - Output from the forward pass is a tensor of probabilities 
         - at = u(st|theta_u) + Nt
    
        Inputs:
            state: 
        Returns:
            action: Max action on the action probabilities (Highest Likely Action/Best Action)

        Reference:
        '''
        action = self.actor_network.forward(state) # output from the actor network is a probability distrobution over the action space
        # Add noise
        action = self.add_noise(action.detach().numpy())
        return torch.tensor(action)
    
    def update_critic(self, states: Tensor, actions: Tensor, next_states: Tensor, rewards: Tensor, terminateds: Tensor):
        next_actions = self.target_actor.forward(next_states).detach()
        target_critic_1_q_value = self.target_critic_q_network_1.forward(next_states, next_actions).detach()
        target_critic_2_q_value = self.target_critic_q_network_2.forward(next_states, next_actions).detach()
        min_q_value = torch.min(target_critic_1_q_value, target_critic_2_q_value).squeeze(1)
        target_q_value = ((min_q_value * self.gamma * (1 - terminateds)) + rewards).unsqueeze(0).t()

        critic_1_q_value = self.critic_q_network_1.forward(states, actions)
        critic_2_q_value = self.critic_q_network_2.forward(states, actions)

        critic_loss = f.mse_loss(critic_1_q_value, target_q_value) + f.mse_loss(critic_2_q_value, target_q_value)

        self.critic_1_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_losses.append(critic_loss.detach().numpy())
    
    def update_actor(self, states: Tensor):
        actions = self.actor_network.forward(states)

        actor_loss = -self.critic_q_network_1.forward(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_losses.append(actor_loss.detach().numpy())
    
    def update_targets(self):
        '''
        Function to soft update the target networks and copy values based proportionally on tau parameter

        Reference:
        '''
        # Soft Update target actor network
        for target_param, param in zip(self.target_actor.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # Soft Update target critic network
        for target_param, param in zip(self.target_critic_q_network_1.parameters(), self.critic_q_network_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_q_network_2.parameters(), self.critic_q_network_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def train(self, episodes):
        for episode in range(episodes):

            # Reset Environment and get state
            state, info = self.env.reset()
            
            state = torch.from_numpy(state).float().ravel()
            terminated, truncated = False, False
            step = 0
            cumulative_reward = 0
            while not terminated and not truncated:
                # Select action
                action = self.get_action(state)

                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.from_numpy(next_state).float().ravel()

                cumulative_reward += reward

                # Store transition
                self.buffer.add(state, action, reward, next_state, terminated)

                if (self.buffer.get_len() > self.batch_size) and (step % self.network_update_rate == 0):
                    # Increment Iterator
                    self.iterator += 1 
                    # Sample a random minibatch of N transitions 
                    mini_batch = self.buffer.sample(self.batch_size)
                    states, actions, rewards, next_states, terminateds = zip(*mini_batch)
                    states = torch.tensor(np.array(states), dtype=torch.float32)
                    actions = torch.tensor(np.array(actions), dtype=torch.float32)
                    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
                    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                    terminateds = torch.tensor(np.array(terminateds), dtype=torch.float32)

                    # Update Critic by minimizing loss
                    self.update_critic(states, actions, next_states, rewards, terminateds)
                    if self.iterator % self.actor_update_frequency == 0:
                        # Update the actor policy using the sampled policy gradient
                        self.update_actor(states)
                        # Update the target networks
                        self.update_targets()

                # Step state to next state 
                state = next_state
                step += 1

            self.episode_rewards.append(cumulative_reward)
            if(episode % 10 == 0):
                if type(cumulative_reward) == Tensor:
                    cumulative_reward = cumulative_reward.item()
                print(f"Episode {episode} reward: {round(cumulative_reward, ndigits=2)}")

    def test(self, episodes):
        for episode in range(episodes):
            # Reset Environment and get state
            state, info = self.env.reset()
            
            state = torch.from_numpy(state).float().ravel()
            terminated, truncated = False, False
            step = 0
            cumulative_reward = 0
            while not terminated and not truncated:
                # Select action
                action = self.get_action(state)

                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.from_numpy(next_state).float().ravel()
                cumulative_reward += reward

                # Step state to next state 
                # self.env.render()
                state = next_state
                step += 1
            self.test_rewards.append(cumulative_reward)
    
    def visualize(self, episodes):
        for episode in range(episodes):
            # Reset Environment and get state
            state, info = self.env.reset()
            
            state = torch.from_numpy(state).float().ravel()
            terminated, truncated = False, False
            step = 0
            cumulative_reward = 0
            while not terminated and not truncated:
                # Select action
                action = self.get_action(state).detach().numpy()

                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.from_numpy(next_state).float().ravel()
                cumulative_reward += reward

                # Step state to next state 
                self.env.render()
                state = next_state
                step += 1
            self.test_rewards.append(cumulative_reward)
        self.env.close()

    def save_model_weights(self, environment_name, agent_name):
        ''' 
        Saves model weights to an environment

        Inputs:
            environment_name: Name of the environment being trained on
        '''
        folder_location = f'/{environment_name}/{agent_name}/'
        path = os.getcwd() + folder_location
        os.makedirs(path, exist_ok=True)
        actor_net = f'{environment_name}_{agent_name}_actor.pth'
        critic_1_net = f'{environment_name}_{agent_name}_critic1.pth'
        critic_2_net = f'{environment_name}_{agent_name}_critic2.pth'
        torch.save(self.actor_network.state_dict(), path + actor_net)
        torch.save(self.critic_q_network_1.state_dict(), path + critic_1_net)
        torch.save(self.critic_q_network_2.state_dict(), path + critic_2_net)

    def load_model_weights(self, environment_name, agent_name):
        path = os.getcwd() + f'/{environment_name}/{agent_name}/'
        actor_net = f'{environment_name}_{agent_name}_actor.pth'
        critic_1_net = f'{environment_name}_{agent_name}_critic1.pth'
        critic_2_net = f'{environment_name}_{agent_name}_critic2.pth'
        self.actor_network.load_state_dict(torch.load(path + actor_net))
        self.critic_q_network_1.load_state_dict(torch.load(path + critic_1_net))
        self.critic_q_network_2.load_state_dict(torch.load(path + critic_2_net))
    
    def save_plot_data(self, environment_name, agent_name, data_name, data):
        path = os.getcwd() + f'/{environment_name}/{agent_name}/'
        file_name = path + f'{environment_name}_{agent_name}_{data_name}.pkl'
        os.makedirs(path, exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

        print(f"{agent_name} Results saved to '{file_name}'")

    def load_plot_data(self, environment_name, agent_name, data_name):
        path = os.getcwd() + f'/{environment_name}/{agent_name}/'
        file_name = path + f'{environment_name}_{agent_name}_{data_name}.pkl'
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    def plot_pickle_files(self, pickle_file_path_list):
        for pickle_file_path in pickle_file_path_list:
            reward_data = self.load_plot_data(pickle_file_path[0], pickle_file_path[1], pickle_file_path[2])
            plt.plot(reward_data)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('All Reward Plots')
        plt.grid(True)
        plt.show()