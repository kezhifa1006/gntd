import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from sec_order.adamw import Adamw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atanh(x):
	one_plus_x = (1 + x).clamp(min=1e-7)
	one_minus_x = (1 - x).clamp(min=1e-7)
	return 0.5*torch.log(one_plus_x/ one_minus_x)

class RegularActor(nn.Module):
	"""A probabilistic actor which does regular stochastic mapping of actions from states"""
	def __init__(self, state_dim, action_dim, max_action,):
		super(RegularActor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.mean = nn.Linear(256, action_dim)
		self.log_std = nn.Linear(256, action_dim)
		self.max_action = max_action
	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean_a = self.mean(a)
		log_std_a = self.log_std(a)
		
		std_a = torch.exp(log_std_a)
		z = mean_a + std_a * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size()))).to(device) 
		return self.max_action * torch.tanh(z)

	def sample_multiple(self, state, num_sample=10):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean_a = self.mean(a)
		log_std_a = self.log_std(a)
		
		std_a = torch.exp(log_std_a)
		# This trick stabilizes learning (clipping gaussian to a smaller range)
		z = mean_a.unsqueeze(1) +\
			 std_a.unsqueeze(1) * torch.FloatTensor(np.random.normal(0, 1, size=(std_a.size(0), num_sample, std_a.size(1)))).to(device).clamp(-0.5, 0.5)
		return self.max_action * torch.tanh(z), z 

	def log_pis(self, state, action=None, raw_action=None):
		"""Get log pis for the model."""
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean_a = self.mean(a)
		log_std_a = self.log_std(a)
		std_a = torch.exp(log_std_a)
		normal_dist = td.Normal(loc=mean_a, scale=std_a, validate_args=True)
		if raw_action is None:
			raw_action = atanh(action)
		else:
			action = torch.tanh(raw_action)
		log_normal = normal_dist.log_prob(raw_action)
		log_pis = log_normal.sum(-1)
		log_pis = log_pis - (1.0 - action**2).clamp(min=1e-6).log().sum(-1)
		return log_pis


class EnsembleCritic(nn.Module):
	""" Critic which does have a network of 4 Q-functions"""
	def __init__(self, num_qs, state_dim, action_dim):
		super(EnsembleCritic, self).__init__()
		
		self.num_qs = num_qs

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

		# self.l7 = nn.Linear(state_dim + action_dim, 400)
		# self.l8 = nn.Linear(400, 300)
		# self.l9 = nn.Linear(300, 1)

		# self.l10 = nn.Linear(state_dim + action_dim, 400)
		# self.l11 = nn.Linear(400, 300)
		# self.l12 = nn.Linear(300, 1)

	def forward(self, state, action, with_var=False):
		all_qs = []
		
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		# q3 = F.relu(self.l7(torch.cat([state, action], 1)))
		# q3 = F.relu(self.l8(q3))
		# q3 = self.l9(q3)

		# q4 = F.relu(self.l10(torch.cat([state, action], 1)))
		# q4 = F.relu(self.l11(q4))
		# q4 = self.l12(q4)

		all_qs = torch.cat(
			[q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
		if with_var:
			std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
			return all_qs, std_q
		return all_qs

	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
	
	def q_all(self, state, action, with_var=False):
		all_qs = []
		
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		# q3 = F.relu(self.l7(torch.cat([state, action], 1)))
		# q3 = F.relu(self.l8(q3))
		# q3 = self.l9(q3)

		# q4 = F.relu(self.l10(torch.cat([state, action], 1)))
		# q4 = F.relu(self.l11(q4))
		# q4 = self.l12(q4)

		all_qs = torch.cat(
			[q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
		if with_var:
			std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
			return all_qs, std_q
			# std_q = torch.std(all_qs, dim=0, keepdim=True, unbiased=False)
			# return torch.cat([all_qs, std_q], 1)
		return all_qs

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	"""VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""
	def __init__(self, state_dim, action_dim, latent_dim, max_action):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device) 
		
		u = self.decode(state, z)

		return u, mean, std
	
	def decode_softplus(self, state, z=None):
		if z is None:
			z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
		
		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		
	def decode(self, state, z=None):
		if z is None:
				z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
	
	def decode_bc(self, state, z=None):
		if z is None:
				z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))

	def decode_bc_test(self, state, z=None):
		if z is None:
				z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
	
	def decode_multiple(self, state, z=None, num_decode=10):
		"""Decode 10 samples atleast"""
		if z is None:
			z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).to(device).clamp(-0.5, 0.5)

		a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a)), self.d3(a)

class BEAR(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, lr=3e-4, 
				 num_qs=2, delta_conf=0.1, use_bootstrap=False, version=0, lambda_=0.4,
				 threshold=0.05, mode='fix', num_samples_match=10, mmd_sigma=10.0,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5,   
				 lagrange_thresh=10.0, use_kl=False, use_ensemble=True, kernel_type='laplacian'):
		self.lr = lr
		latent_dim = action_dim * 2
		self.actor = RegularActor(state_dim, action_dim, max_action).to(device)
		self.actor_target = RegularActor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		# self.actor_optimizer = Adamw(self.actor.parameters(), lr=self.lr)

		self.critic = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
		self.critic_target = EnsembleCritic(num_qs, state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
		# self.critic_optimizer = Adamw(self.critic.parameters(), lr=lr)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr) 
		# self.vae_optimizer = Adamw(self.vae.parameters(), lr=lr) 

		self.discount = discount
		self.tau = tau
		self.max_action = max_action
		self.action_dim = action_dim
		self.delta_conf = delta_conf
		self.use_bootstrap = use_bootstrap
		self.version = version
		self._lambda = lambda_
		self.threshold = threshold
		self.mode = mode
		self.num_qs = num_qs
		self.num_samples_match = num_samples_match
		self.mmd_sigma = mmd_sigma
		self.lagrange_thresh = lagrange_thresh
		self.use_kl = use_kl
		self.use_ensemble = use_ensemble
		self.kernel_type = kernel_type
		
		if self.mode == 'auto':
			# Use lagrange multipliers on the constraint if set to auto mode 
			# for the purpose of maintaing support matching at all times
			self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
			self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2,], lr=1e-3)

		self.step = 0
		self.mask = np.array([])
		

	def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
		"""MMD constraint with Laplacian kernel for support matching"""
		# sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
		diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
		diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

		diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
		diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

		diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
		diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

		overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
		return overall_loss
	
	def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
		"""MMD constraint with Gaussian Kernel support matching"""
		# sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
		diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
		diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

		diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
		diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1, 2))

		diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
		diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1)/(2.0 * sigma)).exp(), dim=(1,2))

		overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
		return overall_loss

	def kl_loss(self, samples1, state, sigma=0.2):
		"""We just do likelihood, we make sure that the policy is close to the
		   data in terms of the KL."""
		state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
		samples1_reshape = samples1.view(-1, samples1.size(-1))
		samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
		samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
		return (-samples1_log_prob).mean(1)
	
	def entropy_loss(self, samples1, state, sigma=0.2):
		state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
		samples1_reshape = samples1.view(-1, samples1.size(-1))
		samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
		samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
		# print (samples1_log_prob.min(), samples1_log_prob.max())
		samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
		return (samples1_prob).mean(1)
	
	def select_action(self, state, evaluation=False):	  
		"""When running the actor, we just select action based on the max of the Q-function computed over
			samples from the policy -- which biases things to support."""
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
			action = self.actor(state)
			q1 = self.critic.q1(state, action)
			ind = q1.max(0)[1]
		return action[ind].cpu().data.numpy().flatten()
	
	def train(self, replay_buffer, batch_size=256):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)


		if self.step == 0 and self.use_bootstrap == True:
			self.mask = np.random.binomial(n=1, size=(replay_buffer.size, self.num_qs, ), p=0.8)
			self.mask = torch.Tensor(self.mask).to(device)

		self.step = self.step + 1
		# Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
		recon, mean, std = self.vae(state, action)
		recon_loss = F.mse_loss(recon, action)
		KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
		vae_loss = recon_loss + 0.5 * KL_loss

		self.vae_optimizer.zero_grad()
		vae_loss.backward()
		self.vae_optimizer.step()

		# Critic Training: In this step, we explicitly compute the actions 
		with torch.no_grad():
			# Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
			state_rep =  torch.repeat_interleave(next_state, 10, 0)
			
			# Compute value of perturbed actions sampled from the VAE
			target_Qs = self.critic_target(state_rep, self.actor_target(state_rep))

			# Soft Clipped Double Q-learning 
			target_Q = 0.75 * target_Qs.min(0)[0] + 0.25 * target_Qs.max(0)[0]
			target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
			target_Q = reward + not_done * self.discount * target_Q

		current_Qs = self.critic(state, action, with_var=False)
		# print(self.mask.shape)
		# print(F.mse_loss(current_Qs[0], target_Q, reduction='none').shape)
		critic_loss_ = torch.reshape(0.5*(F.mse_loss(current_Qs[0], target_Q, reduction='none') + F.mse_loss(current_Qs[1], target_Q, reduction='none')), [-1]) #+ F.mse_loss(current_Qs[2], target_Q) + F.mse_loss(current_Qs[3], target_Q)
		critic_loss = critic_loss_.mean()
		avg = (0.5*(torch.abs(current_Qs[0]-target_Q) + torch.abs(current_Qs[1]-target_Q))).mean().detach().item()

		q_val = (0.5*(current_Qs[0] + current_Qs[1])).cpu().detach().mean().item()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Action Training
		# If you take less samples (but not too less, else it becomes statistically inefficient), it is closer to a uniform support set matching
		num_samples = self.num_samples_match
		sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state, num_decode=num_samples)  # B x N x d
		actor_actions, raw_actor_actions = self.actor.sample_multiple(state, num_samples)#  num)

		# MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
		if self.use_kl:
			mmd_loss = self.kl_loss(raw_sampled_actions, state)
		else:
			if self.kernel_type == 'gaussian':
				mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
			else:
				mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

		action_divergence = ((sampled_actions - actor_actions)**2).sum(-1)
		raw_action_divergence = ((raw_sampled_actions - raw_actor_actions)**2).sum(-1)

		# Update through TD3 style
		# critic_qs, std_q = self.critic.q_all(state, actor_actions[:, 0, :], with_var=True)
		critic_qs = self.critic.q_all(state.unsqueeze(0).repeat(num_samples, 1, 1).view(num_samples*state.size(0), state.size(1)), actor_actions.permute(1, 0, 2).contiguous().view(num_samples*actor_actions.size(0), actor_actions.size(2)))
		critic_qs = critic_qs.view(self.num_qs, num_samples, actor_actions.size(0), 1)
		critic_qs = critic_qs.mean(1)
		std_q = torch.std(critic_qs, dim=0, keepdim=False, unbiased=False)

		if not self.use_ensemble:
			std_q = torch.zeros_like(std_q).to(device)
			
		if self.version == '0':
			critic_qs = critic_qs.min(0)[0]
		elif self.version == '1':
			critic_qs = critic_qs.max(0)[0]
		elif self.version == '2':
			critic_qs = critic_qs.mean(0)

		# We do support matching with a warmstart which happens to be reasonable around epoch 20 during training
		if self.step >= 20: 
			if self.mode == 'auto':
				actor_loss = (-critic_qs +\
					self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
					self.log_lagrange2.exp() * mmd_loss).mean()
			else:
				actor_loss = (-critic_qs +\
					self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q +\
					100.0*mmd_loss).mean()	  # This coefficient is hardcoded, and is different for different tasks. I would suggest using auto, as that is the one used in the paper and works better.
		else:
			if self.mode == 'auto':
				actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
			else:
				actor_loss = 100.0*mmd_loss.mean()

		std_loss = self._lambda*(np.sqrt((1 - self.delta_conf)/self.delta_conf)) * std_q.detach() 

		self.actor_optimizer.zero_grad()
		if self.mode =='auto':
			actor_loss.backward(retain_graph=True)
		else:
			actor_loss.backward()
		# torch.nn.utils.clip_grad_norm(self.actor.parameters(), 10.0)
		self.actor_optimizer.step()

		# Threshold for the lagrange multiplier
		thresh = 0.05
		if self.use_kl:
			thresh = -2.0

		if self.mode == 'auto':
			lagrange_loss = (-critic_qs +\
					self._lambda * (np.sqrt((1 - self.delta_conf)/self.delta_conf)) * (std_q) +\
					self.log_lagrange2.exp() * (mmd_loss - thresh)).mean()

			self.lagrange2_opt.zero_grad()
			(-lagrange_loss).backward()
			# self.lagrange1_opt.step()
			self.lagrange2_opt.step() 
			self.log_lagrange2.data.clamp_(min=-5.0, max=self.lagrange_thresh)   
		
		# Update Target Networks 
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
		return avg, q_val


def weighted_mse_loss(inputs, target, weights):
	return torch.mean(weights * (inputs - target)**2)


