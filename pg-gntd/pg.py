import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sec_order.adamw import Adamw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space=None):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400, 300)

        self.mean_linear = nn.Linear(300, action_dim)
        self.log_std_linear = nn.Linear(300, action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class PGTD(object):
    def __init__(self, state_dim, action_dim, max_action, action_space, lr, discount=0.99, tau=0.02, alpha=0.2, policy_freq=10, ql_type='dqn'):
        
        self.actor = Actor(state_dim, action_dim, action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # relative parameters
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.total_it = 0
        self.policy_freq = policy_freq
        
        self.ql_type = ql_type


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else: 
            _, _, action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1
        # sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf_next_target = self.critic_target(next_state, next_state_action).detach()
            # qf_next_target = qf_next_target - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self.discount * (qf_next_target)
            
            true_next_q_value = self.critic(next_state, next_state_action).detach()
            true_next_q_value = reward + not_done * self.discount * (true_next_q_value)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf = self.critic(state, action) 

        # update Q
        true_qf_loss = torch.abs(qf-true_next_q_value).mean()
        self.critic_optimizer.zero_grad()
        # Compute critic loss
        qf_loss = F.mse_loss(qf, next_q_value)
        qf_loss.backward()
        self.critic_optimizer.step()


        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            act, log_pi, _ = self.actor.sample(state)

            qf_act = self.critic(state, act)

            actor_loss = ((self.alpha * log_pi) - qf_act).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Update the frozen target models
        if self.ql_type == 'dqn':
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        elif self.ql_type == 'td' and self.total_it % 25 == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(param.data)

        return true_qf_loss.detach().cpu().item(), qf.mean().cpu().detach().item()
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
