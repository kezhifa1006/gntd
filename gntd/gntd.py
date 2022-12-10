from optparse import Option
from pyexpat import model
from tkinter import NE
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
# import sec_order
from sec_order.td_kfac import TDKFAC
from sec_order.adamw import Adamw
from utils import ReplayBuffer


class NET(nn.Module):
    def __init__(self, state_size, action_size, layer_size=256):
        super(NET, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.f1 = nn.Linear(self.input_shape[0], layer_size)
        self.f2 = nn.Linear(layer_size, layer_size)
        self.f3 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        x = torch.relu(self.f1(input))
        x = torch.relu(self.f2(x))
        res = self.f3(x)
        
        return res


class Agent_KFAC():
    def __init__(self, state_size, action_size, lr, damping, pi, constraint_norm, batch_averaged, momentum=False,
                hidden_size=256, device="cpu", option='dqn', buffer_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = 0.99
        self.tau = 2e-2
        self.fqi_num = 10
        self.step = 0
        self.option = option
        
        self.network = NET(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_net = copy.deepcopy(self.network)

        self.tmp_net = copy.deepcopy(self.network)
        
        if momentum:
            # self.optimizer = Adamw(params=self.network.parameters(), lr=lr, betas=(0.9, 0.999)) 
            self.optimizer = optim.Adam(params=self.network.parameters(), lr=lr)
        else:    
            # self.optimizer = Adamw(params=self.network.parameters(), lr=lr, betas=(0, 0.999))
            self.optimizer = optim.Adam(params=self.network.parameters(), lr=lr)
        self.tmp_optimizer = TDKFAC(model=self.tmp_net, lr=lr, damping=damping, 
                                    pi=pi, constraint_norm=constraint_norm)
        self.batch_averaged = batch_averaged

        self.replay_buffer = ReplayBuffer(buffer_size, device)
    
    def append_sample(self, state, action, next_state, reward, done_bool):
        self.replay_buffer.add(state, action, next_state, reward, done_bool)       
    
    def get_action(self, state, epsilon):
        self.tmp_optimizer.stat_decay = min(0.95, max(0.1, 1-epsilon))
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            # self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            # self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action
        
    def learn(self, batch_size):
        self.step += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_tmp = self.tmp_net(states).gather(1, actions)

        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(1, actions)
        q_avg = Q_expected.detach().mean().cpu().item()

        dqn_loss = F.mse_loss(Q_expected, Q_targets)
        # dqn_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        avg_loss = (torch.abs(Q_expected-Q_targets).detach()).mean().cpu().item() 
        self.optimizer.zero_grad()
        dqn_loss.backward()
        clip_grad_norm_(self.network.parameters(), 10)

        # tmp
        if self.batch_averaged:
            loss_tmp = torch.mean(Q_tmp)
        else:
            loss_tmp = torch.sum(Q_tmp)
        self.tmp_optimizer.zero_grad()
        loss_tmp.backward()
        
        self.tmp_optimizer.step(optimizer=self.optimizer)
        self.optimizer.step()
        self.hard_update(self.network, self.tmp_net)
        
        # updates = self.optimizer.step_()
        # self.tmp_optimizer.step(optimizer=self.optimizer, updates_tmp=updates)
        # self.hard_update(self.tmp_net, self.network)

        # ------------------- update target network ------------------- #
        if self.option == 'dqn':
            self.soft_update(self.network, self.target_net)
        elif self.option == 'td':
            self.hard_update(self.network, self.target_net)
        elif self.option == 'fqi':
            if int(self.step) % int(self.fqi_num) == 0: self.hard_update(self.network, self.target_net)

        return avg_loss, q_avg
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)