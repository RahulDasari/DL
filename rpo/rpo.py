import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import tyro
from torch.distributions.normal import Normal
# from torch.utils.tensorboard import SummaryWriter


class Network(nn.Module) :
    def __init__(self, envs, alpha):
        super().__init__()
        self.alpha = alpha
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64 , 64),
            nn.Tanh(),
            nn.Linear(64,1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64 , 64),
            nn.Tanh(),
            nn.Linear(64 , np.prod(envs.single_action_space.shape)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        # for layer in self.actor_mean:
        #     if isinstance(layer, nn.Linear):
        #         layer.weight.data = layer.weight.data.type(torch.float64)
        #         layer.bias.data = layer.bias.data.type(torch.float64)
            
    def get_value(self, x):
        return self.critic(x)
    
    def get_probs(self,x):
        return self.actor_mean(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.alpha, self.alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class RPO_Agent :

    def storage_setup(self , num_steps , num_envs , observation_space , action_space) :

        obs = torch.zeros((num_steps, num_envs) + observation_space)
        actions = torch.zeros((num_steps, num_envs) + action_space)
        logprobs = torch.zeros((num_steps, num_envs))
        rewards = torch.zeros((num_steps, num_envs))
        dones = torch.zeros((num_steps, num_envs))
        values = torch.zeros((num_steps, num_envs))

        return obs , actions , logprobs , rewards , dones , values
    
    def get_advantages(self , num_steps , next_done , next_value , dones , values , rewards , gamma , gae_lambda) :
        
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        return advantages
    
    def flatten_batch(self , obs , logprobs , actions ,advantages , returns , values , observation_space_shape , action_space_shape):

        b_obs = obs.reshape((-1,) + observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        return b_obs , b_logprobs , b_actions ,b_advantages , b_returns , b_values


    def get_policy_loss(self , logratio , ratio , b_advantages , mb_inds , clip_coef , clipfracs) :

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds]

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        return pg_loss
    
    def get_value_loss(self , clip_vloss , b_returns , mb_inds , b_values , clip_coef , newvalue):

        newvalue = newvalue.view(-1)
        if clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        return v_loss