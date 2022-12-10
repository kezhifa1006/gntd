import numpy as np
import torch
import gym
import argparse
import os

import utils
import pg
import pg_gntd


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD")                  # Policy name (TD3, DDPG, GNTD3, GNDDPG, LSDDPG, LSTD3)
    parser.add_argument("--env", default="HalfCheetah-v3")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=25e5, type=int)   # Max time steps to run environment
    parser.add_argument("--policy_freq", default=25, type=int)      # Frequency of delayed policy updates
    parser.add_argument("--lr", default=0.0003, type=float)         # Learning rate for Q learning
    parser.add_argument("--ql_type", default="dqn", choices=['dqn', 'td'])  # Q learning type
    parser.add_argument("--alpha", default=0.5, type=float)         # Choice of alpha for KL divergence w.r.t. policy pi
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.02, type=float)          # Target network update rate
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--save_mat", action="store_true")          # Save information mat
    # KFAC option
    parser.add_argument("--momentum", action="store_true")          # Momentum for GNTD 
    parser.add_argument("--damping", default=0.01, type=float)      # Damping for GNTD
    parser.add_argument("--pi", action="store_true")                 # Pi for GNTD  
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.ql_type}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, QLType: {args.ql_type}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "action_space": env.action_space,
        "policy_freq": args.policy_freq,
        "discount": args.discount,
        "tau": args.tau,
        "lr": args.lr,
        "alpha": args.alpha,
        "ql_type": args.ql_type,
    }

    # Initialize policy
    if args.policy == "TD":
        policy = pg.PGTD(**kwargs)
    elif args.policy == 'GNTD':
        print(f"KFAC Option: lr {args.lr} | damping {args.damping} | pi {args.pi}")
        print("---------------------------------------")
        # KFAC
        kwargs["momentum"] = args.momentum
        kwargs["damping"] = args.damping
        kwargs["pi"] = args.pi
        policy = pg_gntd.PGGNTD(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    eval_policy(policy, args.env, args.seed)
    losses = []
    q_vals = []
    rewards = []

    loss = 0
    avg = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            loss, avg = policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            if t >= args.start_timesteps:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Q_val: {avg} Loss: {loss}")
            else:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            rewards.append(eval_policy(policy, args.env, args.seed))
            losses.append(loss)
            q_vals.append(avg)
            if args.save_model: policy.save(f"./models/{file_name}")
            if args.save_mat:
                np.save(f"./results/{file_name}_rewards", rewards)
                np.save(f"./results/{file_name}_loss", losses)
                np.save(f"./results/{file_name}_qval", q_vals)
