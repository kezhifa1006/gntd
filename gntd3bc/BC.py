from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sec_order.adamw import Adamw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class BC(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, 
            policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5, lr=3e-4):
		self.device = device
		self.lr = lr 
		self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		# self.actor_optimizer = Adamw(self.actor.parameters(), lr=self.lr)

		self.ptr = 0


	def select_action(self, state, evaluate=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		avg = 0

		# sample replay buffer
		a_pred = self.actor(state)
		loss = F.mse_loss(a_pred, action)
		avg += loss.item()
		self.actor_optimizer.zero_grad()
		loss.backward()
		self.actor_optimizer.step()

		return avg, -1

	
	def save(self, filename):
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

