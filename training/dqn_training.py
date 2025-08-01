from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from environment.custom_env import DiscreteRwandaFarmEnv
import os
import logging
import numpy as np
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_dqn():
    logging.info("Starting DQN training")
    env = DiscreteRwandaFarmEnv()
    check_env(env)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1.5e-3,
        buffer_size=10000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=1
    )
    total_timesteps = 500000
    start_time = time.time()
    losses = []
    
    # Callback to log loss
    def callback(_locals, _globals):
        if 'loss' in _locals:
            losses.append(_locals['loss'])
        return True
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    convergence_time = time.time() - start_time
    logging.info(f"DQN Convergence Time: {convergence_time} seconds")
    
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
            logging.info(f"DQN Episode {ep+1}/{episodes} Reward={episode_reward}, Steps={steps}")
    avg_reward = np.mean(rewards)
    logging.info(f"DQN Average Reward: {avg_reward}, Average Steps: {np.mean([s for s in range(90) if not done])}")
    
    os.makedirs("models/dqn", exist_ok=True)
    model.save("models/dqn/dqn_farm")
    
    # Plot DQN objective function
    plt.figure()
    plt.plot(losses)
    plt.title("DQN Objective Function")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.savefig("DQN_objective_plot.png")
    plt.close()
    
    return model, rewards

if __name__ == "__main__":
    train_dqn()