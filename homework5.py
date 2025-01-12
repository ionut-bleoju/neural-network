import flappy_bird_gymnasium
import gymnasium
import torch
import itertools
import os
import random
from datetime import datetime
from collections import deque
import random
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        Q = self.output(x)

        return Q
    

class MemoryReplay:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearningAgent:
    def __init__(self):
        self.set_name = "flappy_bird"
        self.memory_size = 100000
        self.batch_size = 64
        self.epsilon_init = 1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.sync_freq = 10
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.set_name}.pt")

    def run(self, is_train: bool, render: bool):
        if is_train:
            start_time = datetime.now()
            print(f"{start_time.strftime(DATE_FORMAT)}: Training starting...")


        env = gymnasium.make(
            "FlappyBird-v0", render_mode="human" if render else None, use_lidar=False
        )

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_train:
            epsilon = self.epsilon_init

            replay = MemoryReplay(self.memory_size)

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate
            )

            epsilon_history = []

            step = 0

            best_reward = -99999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_train and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_train:
                    replay.add((state, action, new_state, reward, terminated))
                    step += 1
                    if episode_reward > best_reward:
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                        print(log_message)
                        best_reward = episode_reward
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                state = new_state

            if is_train:
                if len(replay) > self.batch_size:
                    batch = replay.sample(self.batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step > self.sync_freq:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step = 0


    def optimize(self, batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.discount_factor
                * target_dqn(new_states).max(dim=1)[0]
            )
        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = QLearningAgent()
    agent.run(is_train=False, render=True)
