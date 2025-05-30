import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
import json
import os

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools
import signal

import flappy_bird_gymnasium

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info and metrics
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': generate plots as files
matplotlib.use('Agg')

# Device selection: use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Frame-skip wrapper to accelerate training
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                done = True
                break
        return obs, total_reward, terminated, truncated, info

class Agent():
    def __init__(self, hyperparameter_set, checkpoint_interval):
        # load hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_sets = yaml.safe_load(file)
            self.params = all_sets[hyperparameter_set]
        self.set_name = hyperparameter_set
        # extract optional frame skip
        self.frame_skip = self.params.get('frame_skip', 1)
        # paths
        self.log_path      = os.path.join(RUNS_DIR, f'{self.set_name}.log')
        self.model_latest  = os.path.join(RUNS_DIR, f'{self.set_name}.pt')
        self.graph_path    = os.path.join(RUNS_DIR, f'{self.set_name}.png')
        self.checkpoint_interval = checkpoint_interval
        # save params snapshot
        with open(os.path.join(RUNS_DIR, f'{self.set_name}_params.json'), 'w') as f:
            json.dump(self.params, f, indent=2)
        # hyperparams
        p = self.params
        self.env_id             = p['env_id']
        self.lr                 = p['learning_rate_a']
        self.gamma              = p['discount_factor_g']
        self.sync_rate          = p['network_sync_rate']
        self.memory_size        = p['replay_memory_size']
        self.batch_size         = p['mini_batch_size']
        self.epsilon            = p['epsilon_init']
        self.epsilon_decay      = p['epsilon_decay']
        self.epsilon_min        = p['epsilon_min']
        self.stop_on            = p['stop_on_reward']
        self.hidden_size        = p['fc1_nodes']
        self.env_kwargs         = p.get('env_make_params', {})
        self.double_q           = p['enable_double_dqn']
        self.dueling            = p['enable_dueling_dqn']

        self.loss_fn            = nn.MSELoss()
        self.optimizer          = None
        # metrics
        self.rewards           = []
        self.epsilon_history   = []
        self.loss_history      = []
        # graceful shutdown
        signal.signal(signal.SIGINT, self._handle_sigint)
        self._stop_requested   = False

    def _handle_sigint(self, signum, frame):
        print("\nKeyboard interrupt received. Stopping after current episode...")
        self._stop_requested = True

    def run(self, is_training=True, render=False, checkpoint_file=None):
        # logging start
        if is_training:
            start = datetime.now()
            with open(self.log_path, 'w') as log:
                log.write(f"{start.strftime(DATE_FORMAT)}: Training started\n")
        # create and wrap env
        raw_env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_kwargs)
        env = FrameSkip(raw_env, self.frame_skip)
        n_actions = env.action_space.n
        state_dim = env.observation_space.shape[0]
        # build networks
        policy = DQN(state_dim, n_actions, self.hidden_size, self.dueling).to(device)
        if is_training:
            target = DQN(state_dim, n_actions, self.hidden_size, self.dueling).to(device)
            target.load_state_dict(policy.state_dict())
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
            mem = ReplayMemory(self.memory_size)
            step_count = 0
            best = -1e9
        else:
            # load checkpoint or latest
            model_to_load = checkpoint_file or self.model_latest
            policy.load_state_dict(torch.load(model_to_load, map_location=device))
            policy.to(device).eval()

        try:
            for ep in itertools.count():
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float, device=device)
                done = False
                ep_reward = 0.0
                while not done and ep_reward < self.stop_on:
                    # ε-greedy
                    if is_training and random.random() < self.epsilon:
                        action = env.action_space.sample()
                        action = torch.tensor(action, dtype=torch.int64, device=device)
                    else:
                        with torch.no_grad():
                            action = policy(state.unsqueeze(0)).squeeze().argmax()
                    obs, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    ep_reward += reward
                    new_state = torch.tensor(obs, dtype=torch.float, device=device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float, device=device)
                    if is_training:
                        mem.append((state, action, new_state, reward_tensor, done))
                        step_count += 1
                    state = new_state
                # after episode
                if is_training:
                    self.rewards.append(ep_reward)
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                    self.epsilon_history.append(self.epsilon)
                    if len(mem) > self.batch_size:
                        loss = self.optimize(mem.sample(self.batch_size), policy, target)
                        self.loss_history.append(loss)
                        if step_count > self.sync_rate:
                            target.load_state_dict(policy.state_dict()); step_count = 0
                    if ep_reward > best:
                        best = ep_reward; torch.save(policy.state_dict(), self.model_latest)
                    if ep % self.checkpoint_interval == 0 and ep>0:
                        cp = os.path.join(RUNS_DIR, f'{self.set_name}_ep{ep}.pt')
                        torch.save(policy.state_dict(), cp)
                        self.save_graph(); self.save_metrics(suffix=f'_ep{ep}')
                if self._stop_requested:
                    break
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            if is_training:
                self.save_graph(); self.save_metrics(); torch.save(policy.state_dict(), self.model_latest)
                print("Training stopped. Artifacts saved.")

    def optimize(self, batch, policy, target):
        states, acts, new_states, rewards, dones = zip(*batch)
        states = torch.stack(states); acts = torch.stack(acts)
        new_states = torch.stack(new_states); rewards = torch.stack(rewards)
        dones = torch.tensor(dones, dtype=torch.float, device=device)
        with torch.no_grad():
            if self.double_q:
                best_next = policy(new_states).argmax(dim=1)
                q_next = target(new_states).gather(1, best_next.unsqueeze(1)).squeeze()
            else:
                q_next = target(new_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * q_next
        current_q = policy(states).gather(1, acts.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()

    def save_graph(self):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        mean_r = [np.mean(self.rewards[max(0,i-99):i+1]) for i in range(len(self.rewards))]
        plt.plot(mean_r); plt.title('Mean Reward')
        plt.subplot(1,2,2)
        plt.plot(self.epsilon_history); plt.title('Epsilon')
        plt.tight_layout(); plt.savefig(self.graph_path); plt.close()

    def save_metrics(self, suffix=''):
        np.savetxt(os.path.join(RUNS_DIR, f'{self.set_name}{suffix}_rewards.csv'), np.array(self.rewards), delimiter=',')
        np.savetxt(os.path.join(RUNS_DIR, f'{self.set_name}{suffix}_epsilon.csv'), np.array(self.epsilon_history), delimiter=',')
        np.savetxt(os.path.join(RUNS_DIR, f'{self.set_name}{suffix}_loss.csv'), np.array(self.loss_history), delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameters')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--render', action='store_true', help='display the game window during evaluation')
    parser.add_argument('--checkpoint-interval', type=int, default=50, help='episodes between checkpoint saves')
    parser.add_argument('--checkpoint-file', type=str, help='path to model checkpoint for evaluation')
    args = parser.parse_args()
    agent = Agent(args.hyperparameters, checkpoint_interval=args.checkpoint_interval)
    agent.run(is_training=args.train, render=args.render, checkpoint_file=args.checkpoint_file)
