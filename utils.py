import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque, namedtuple


class Random():
    def __init__(self, state_size, action_size, device="cpu", buffer_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.replay_buffer = ReplayBuffer(buffer_size, device)

    def append_sample(self, state, action, next_state, reward, done_bool):
        self.replay_buffer.add(state, action, next_state, reward, done_bool) 

    def get_action(self, state, epsilon):
        return random.choices(np.arange(self.action_size), k=1)
    
    def learn(self, batch_size):
        return -1, -1


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.mean = 0
        self.std = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.stack([(e.state-self.mean)/self.std for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([(e.next_state-self.mean)/self.std for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def normalize_states(self, eps = 1e-3):
        states, _, _, next_states, _ = self.sample(len(self.memory))
        self.mean = states.mean(0).cpu().numpy()
        self.std = (states.std(0) + eps).cpu().numpy()
    
    def load(self, save_folder):
        state = np.load(f"{save_folder}_state.npy")
        action = np.load(f"{save_folder}_action.npy")
        reward = np.load(f"{save_folder}_reward.npy")
        next_state = np.load(f"{save_folder}_next_state.npy")
        done = np.load(f"{save_folder}_not_done.npy")

        for s, a, r, n_s, d in zip(state, action, reward, next_state, done):
            self.add(s, a, r, n_s, d)
    
    def save(self, save_folder):
        states = np.stack([e.state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
        next_states = np.stack([e.next_state for e in self.memory if e is not None])
        dones = np.vstack([e.done for e in self.memory if e is not None])

        np.save(f"{save_folder}_state.npy", states)
        np.save(f"{save_folder}_action.npy", actions)
        np.save(f"{save_folder}_next_state.npy", next_states)
        np.save(f"{save_folder}_reward.npy", rewards)
        np.save(f"{save_folder}_not_done.npy", dones)


def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")


if __name__ == "__main__":
    envs = ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    seeds = ['0', '1', '2', '3', '4']
    for env in envs:
        for seed in seeds:
            name1 = f'{env}_{seed}_Expert'
            name2 = f'{env}_{seed}_Random'
            # Load buffer
            buffer1 = ReplayBuffer(100000, 'cpu')
            buffer2 = ReplayBuffer(100000, 'cpu')
            mix_buffer = ReplayBuffer(100000, 'cpu')
            buffer1.load(f"./buffers/{name1}")
            buffer2.load(f"./buffers/{name2}")

            data = random.sample(buffer1.memory, k=int(0.8*len(buffer1)))
            states = np.stack([e.state for e in data if e is not None])
            actions = np.vstack([e.action for e in data if e is not None])
            rewards = np.vstack([e.reward for e in data if e is not None])
            next_states = np.stack([e.next_state for e in data if e is not None])
            dones = np.vstack([e.done for e in data if e is not None])
            for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                mix_buffer.add(s, a, r, n_s, d)

            data = random.sample(buffer2.memory, k=int(0.2*len(buffer2)))
            states = np.stack([e.state for e in data if e is not None])
            actions = np.vstack([e.action for e in data if e is not None])
            rewards = np.vstack([e.reward for e in data if e is not None])
            next_states = np.stack([e.next_state for e in data if e is not None])
            dones = np.vstack([e.done for e in data if e is not None])
            for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                mix_buffer.add(s, a, r, n_s, d)
            
            mix_buffer.save(f'./buffers/{env}_{seed}_Mix')

            

