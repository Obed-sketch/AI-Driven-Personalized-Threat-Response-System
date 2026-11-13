"""Steps Two-Four of the original script: Q-network, training, policies."""
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Two-layer Q-network, identical architecture to the original script."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def rule_based_policy(observation):
    """The A/B baseline from the original script: always recommend 'scan'."""
    return 1


class RLAgent:
    """Adapter that feeds an observation through a trained Q-network
    (the glue class added during the Nov 2026 repair, kept here)."""

    def __init__(self, network):
        self.network = network

    def predict(self, observation):
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)
        )
        with torch.no_grad():
            return int(self.network(obs).argmax().item())


def train_agent(env, episodes=200, seed=42, epsilon=0.1, gamma=0.99, lr=0.001):
    """Train a Q-network on `env` with the original script's simplified
    Q-learning update, plus:
    - a fixed seed (reproducible runs/tests), and
    - epsilon-greedy action selection (with epsilon=0.1). The original
      script always acted greedily, which on this deterministic walk-through
      environment means it only ever updates the single action it already
      prefers; a small exploration rate lets the other actions' values be
      learned too. epsilon=0.0 reproduces the original behaviour.

    Returns (network, episode_rewards)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = QNetwork(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    episode_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            state_tensor = torch.FloatTensor(state)
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    action = int(network(state_tensor).argmax().item())
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            # Simplified Q-learning update, as in the original script.
            target = reward + gamma * network(
                torch.FloatTensor(next_state)
            ).max().item()
            loss = loss_fn(
                network(state_tensor)[action],
                torch.tensor(target, dtype=torch.float32),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            if done:
                break
        episode_rewards.append(total_reward)
    return network, episode_rewards


def evaluate_policy(env, policy):
    """Run one pass over `env`, choosing actions with policy(observation).
    Returns (total_reward, mean_reward_per_step)."""
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action = policy(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    return total_reward, total_reward / max(steps, 1)
