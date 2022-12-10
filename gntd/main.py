from email import policy
from lib2to3.pgen2.token import LBRACE
from os import stat
import gym
import numpy as np
import math
import torch
import argparse
from utils import save, collect_random, Random
import random
import scipy.io as sio
from dqn import Agent
from gntd import Agent_KFAC
from utils import ReplayBuffer

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name, default: CartPole-v1")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes, default: 300")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Maximal training dataset size, default: 100000")
    parser.add_argument("--batch_size", type=int, default=100, help="Training for each batch size, default: 100")
    parser.add_argument("--seed", type=int, default=0, help="Seed, default: 0")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate for Model")
    parser.add_argument("--option", default="dqn", choices=['dqn', 'td', 'fqi'], help="Choice for TD Methods") 
    parser.add_argument("--save_mat", action="store_true", help='Choice for saving details')
    # generate buffer
    parser.add_argument("--policy", default="Expert", choices=['Expert', 'Random', 'Medium-Replay', 'Replay', 'Mix'], help="Choice for TD Methods") 
    parser.add_argument("--save_buffer", action="store_true", help='Choice for saving buffers')
    # offline for gntd, dqn
    parser.add_argument("--train_offline", action="store_true", help="Train Offline, default: False")
    parser.add_argument("--use_gntd", action="store_true", help='Choice for choosing gntd')
    parser.add_argument("--max_timesteps", default=3000, type=int, help="Train Offline, default: 50000")
    # KFAC option
    parser.add_argument("--momentum", action="store_true")         # Momentum for GNTD 
    parser.add_argument("--damping", default=0.5, type=float)     # Damping for GNTD
    parser.add_argument("--pi", action="store_true")     		   # Pi for GNTD  
    args = parser.parse_args()
    return args

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(agent, env_name, seed, eval_episodes=20, replay_buffer=None, output=True):
    eval_env = gym.make(env_name)
    if replay_buffer is not None:
        eval_env.seed(seed)
    else:
        eval_env.seed(seed + 100)

    rewards = torch.zeros(eval_episodes)
    for i in range(eval_episodes):
        reward_ = 0
        state, done = eval_env.reset(), False
        while not done:
            action = agent.get_action(np.array(state), 0)
            next_state, reward, done, _ = eval_env.step(action[0])
            if replay_buffer is not None:
                replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            reward_ += reward
        rewards[i] = reward_

    avg_reward = torch.mean(rewards)

    if output:
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")

    return avg_reward, replay_buffer


# Train online
def online_train(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(args.env)
    
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # For saving files
    setting = f"{args.env}_{args.seed}"
    save_name = f"./results/QL_{setting}_{args.option}"
    
    eps = 1.
    d_eps = 1 - args.min_eps
    steps = 0
    total_steps = 0

    avg_rewards = []
    avg_loss = []
    q_vals = []
    
    if args.policy == 'Replay':
        agent = Agent(state_size=env.observation_space.shape, action_size=env.action_space.n, option='dqn',
                    lr=args.lr, device=device, buffer_size=args.buffer_size)
    elif args.policy == 'Medium-Replay':
        agent = Agent(state_size=env.observation_space.shape, action_size=env.action_space.n, option='td',
                    lr=args.lr, device=device, buffer_size=args.buffer_size)
    elif args.policy == 'Random':
        agent = Random(state_size=env.observation_space.shape, action_size=env.action_space.n, device=device, buffer_size=args.buffer_size)
    
    
    collect_random(env=env, dataset=agent.replay_buffer, num_samples=10000)

    for i in range(1, args.episodes+1):
        state = env.reset()
        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state, epsilon=eps)
            steps += 1
            next_state, reward, done, _ = env.step(action[0])
            agent.append_sample(state, action, reward, next_state, done)
            loss, Q_avg = agent.learn(args.batch_size)
            state = next_state
            rewards += reward
            episode_steps += 1
            eps = max(1 - ((steps*d_eps)/args.eps_frames), args.min_eps)
            if done:
                break   

        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Avg Loss: {} | Q Avg Value: {} | Steps: {}".format(i, rewards, loss, Q_avg, steps,))
        avg_rewards.append(rewards)
        avg_loss.append(loss)
        q_vals.append(Q_avg)

    if args.save_mat:
        np.save(f"{save_name}_avg_rewards.npy", avg_rewards)
        np.save(f"{save_name}_avg_loss.npy", avg_loss)
        np.save(f"{save_name}_q_vals.npy", q_vals)
    
    if args.save_buffer:
        agent.replay_buffer.save(f"./buffers/{setting}_{args.policy}")
        if args.policy == 'Replay':
            replay_buffer = ReplayBuffer(args.buffer_size, device)
            eval_policy(agent, args.env, args.seed, args.episodes, replay_buffer)
            replay_buffer.save(f"./buffers/{setting}_Expert")


# Trains offline
def offline_train(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = gym.make(args.env)    
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    # For saving files
    setting = f"{args.env}_{args.seed}_{args.policy}"

    if args.use_gntd:
        agent = Agent_KFAC(state_size=env.observation_space.shape, action_size=env.action_space.n, lr=args.lr, 
                        damping=args.damping, pi=args.pi, constraint_norm=False, batch_averaged=False,
                        option=args.option, device=device, buffer_size=args.buffer_size)
        save_name = f"./results/Offline_gntd_{args.option}_{setting}"
    else:   
        agent = Agent(state_size=env.observation_space.shape, action_size=env.action_space.n, option=args.option,
                        lr=args.lr, device=device, buffer_size=args.buffer_size)
        save_name = f"./results/Offline_{args.option}_{setting}"
    
    # Load buffer
    agent.replay_buffer.load(f"./buffers/{setting}")

    # agent.replay_buffer.normalize_states()

    avg_rewards = []
    avg_loss = []
    q_vals = []

    training_iters = 0

    while training_iters < args.max_timesteps: 
        training_iters += 1
        avg, q_val = agent.learn(args.batch_size)
        # avg_reward, _ = eval_policy(agent, args.env, args.seed, output=False)
        # print("Episode: {} | Avg Reward: {} | Avg Loss: {} | Q Avg Value: {}".format(training_iters, avg_reward, avg, q_val))
        if training_iters % 30 == 0: 
            avg_reward, _ = eval_policy(agent, args.env, args.seed, output=False)
            avg_loss.append(avg)
            q_vals.append(q_val)
            avg_rewards.append(avg_reward)
            print("Episode: {} | Avg Reward: {} | Avg Loss: {} | Q Avg Value: {}".format(training_iters, avg_reward, avg, q_val))

    if args.save_mat:
        np.save(f"{save_name}_avg_rewards.npy", avg_rewards)
        np.save(f"{save_name}_avg_loss.npy", avg_loss)
        np.save(f"{save_name}_q_vals.npy", q_vals)



if __name__ == "__main__":
    args = get_config()
    if not args.train_offline:
        print("---------------------------------------")
        print(f"Training online. Env {args.env} | Seed: {args.seed} | Option: {args.option}")
        print("---------------------------------------")
        online_train(args)
    else:
        print("---------------------------------------")
        print(f"Training offline. Env {args.env} | Seed: {args.seed} | Use GNTD: {args.use_gntd} | Option: {args.option} | Buffer name: {args.policy}")
        print("---------------------------------------")
        offline_train(args)