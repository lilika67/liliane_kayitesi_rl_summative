from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment.custom_env import RwandaFarmEnv
import os
import logging
import matplotlib.pyplot as plt
from training.dqn_training import train_dqn
import time  # Confirm this line is present

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_reinforce(env, episodes=1000, lr=4e-3, gamma=0.99):
    logging.info("Starting REINFORCE training")
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    rewards_history = []
    entropies = []

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        steps = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state_tensor).squeeze(0)
            dist = torch.distributions.Normal(action_probs, 0.1)
            action = dist.sample().clamp(-1, 1)
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().mean().item()
            entropies.append(entropy)

            state, reward, done, _, _ = env.step(action.detach().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)
            steps += 1

        discounted_rewards = []
        G = 0
        for r in rewards[::-1]:
            G = r + gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        policy_loss = []
        for log_prob, G in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)
        if (episode + 1) % 100 == 0:
            logging.info(
                f"REINFORCE Episode {episode+1}/{episodes} Reward={episode_reward}, Steps={steps}"
            )

    avg_reward = np.mean(rewards_history)
    logging.info(
        f"REINFORCE Average Reward: {avg_reward}, Total Steps: {steps * episodes}"
    )
    return policy, rewards_history, entropies


def train_ppo():
    logging.info("Starting PPO training")
    env = RwandaFarmEnv()
    check_env(env)
    model = PPO(
        "MlpPolicy", env, learning_rate=2e-4, n_steps=2048, clip_range=0.2, verbose=1
    )
    total_timesteps = 500000
    start_time = time.time()  # Now works with import
    entropies = []

    def callback(_locals, _globals):
        if "entropy_loss" in _locals:
            entropies.append(_locals["entropy_loss"].item())
        return True

    model.learn(total_timesteps=total_timesteps, callback=callback)
    convergence_time = time.time() - start_time
    logging.info(f"PPO Convergence Time: {convergence_time} seconds")

    rewards = []
    episodes = 100
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
        rewards.append(episode_reward)
        if (ep + 1) % 10 == 0:
            logging.info(
                f"PPO Episode {ep+1}/{episodes} Reward={episode_reward}, Steps={steps}"
            )
    avg_reward = np.mean(rewards)
    logging.info(
        f"PPO Average Reward: {avg_reward}, Average Steps: {np.mean([s for s in range(90) if not done])}"
    )

    os.makedirs("models/pg", exist_ok=True)
    model.save("models/pg/ppo_farm")
    return model, rewards, entropies


def train_a2c():
    logging.info("Starting A2C training")
    env = RwandaFarmEnv()
    check_env(env)
    model = A2C("MlpPolicy", env, learning_rate=4e-3, n_steps=20, verbose=1)
    total_timesteps = 500000
    start_time = time.time()  # Now works with import
    entropies = []

    def callback(_locals, _globals):
        if "entropy_loss" in _locals:
            entropies.append(_locals["entropy_loss"].item())
        return True

    model.learn(total_timesteps=total_timesteps, callback=callback)
    convergence_time = time.time() - start_time
    logging.info(f"A2C Convergence Time: {convergence_time} seconds")

    rewards = []
    episodes = 100
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
        rewards.append(episode_reward)
        if (ep + 1) % 10 == 0:
            logging.info(
                f"A2C Episode {ep+1}/{episodes} Reward={episode_reward}, Steps={steps}"
            )
    avg_reward = np.mean(rewards)
    logging.info(
        f"A2C Average Reward: {avg_reward}, Average Steps: {np.mean([s for s in range(90) if not done])}"
    )

    os.makedirs("models/pg", exist_ok=True)
    model.save("models/pg/a2c_farm")
    return model, rewards, entropies


def train_all():
    ppo_model, ppo_rewards, ppo_entropies = train_ppo()
    a2c_model, a2c_rewards, a2c_entropies = train_a2c()
    env = RwandaFarmEnv()
    reinforce_policy, reinforce_rewards, reinforce_entropies = train_reinforce(env)
    dqn_model, dqn_rewards = train_dqn()

    os.makedirs("models/pg", exist_ok=True)
    torch.save(reinforce_policy.state_dict(), "models/pg/reinforce_farm.pt")

    # Plot PG entropy
    plt.figure()
    plt.plot(ppo_entropies, label="PPO")
    plt.plot(reinforce_entropies, label="REINFORCE")
    plt.plot(a2c_entropies, label="A2C")
    plt.title("Policy Entropy")
    plt.xlabel("Training Steps")
    plt.ylabel("Entropy")
    plt.legend()
    plt.savefig("PG_entropy_plot.png")
    plt.close()

    return (
        ppo_model,
        a2c_model,
        reinforce_policy,
        dqn_model,
        ppo_rewards,
        a2c_rewards,
        reinforce_rewards,
        dqn_rewards,
    )


if __name__ == "__main__":
    train_all()
