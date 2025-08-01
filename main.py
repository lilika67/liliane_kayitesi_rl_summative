from training.pg_training import train_all
from training.dqn_training import train_dqn
from implementation.rendering import RwandaFarmEnvVisualizer
from environment.custom_env import RwandaFarmEnv
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    env = RwandaFarmEnv()
    visualizer = RwandaFarmEnvVisualizer(env)

    logging.info("Starting model training")
    (
        ppo_model,
        a2c_model,
        reinforce_policy,
        dqn_model,
        ppo_rewards,
        a2c_rewards,
        reinforce_rewards,
        dqn_rewards,
    ) = train_all()

    logging.info("Rendering random agent visualization")
    visualizer.render("random_farm.gif")
    logging.info("Rendering trained agent visualization with PPO")
    visualizer.render("trained_farm.mp4", is_trained=True, model=ppo_model)

    plt.figure()
    plt.plot(ppo_rewards, label="PPO")
    plt.plot(a2c_rewards, label="A2C")
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(reinforce_rewards, label="REINFORCE")
    plt.legend()
    plt.savefig("reward_plot.png")
    plt.close()
    logging.info("Saved reward_plot.png")

    # Collect state and action data for PPO (as an example)
    states = []
    actions = []
    state, _ = env.reset()
    done = False
    while not done and len(states) < 1000:  # Limit to 1000 steps for plotting
        action, _ = ppo_model.predict(state, deterministic=True)
        state, _, done, _, _ = env.step(action)
        states.append(state.copy())
        actions.append(action.copy() if isinstance(action, np.ndarray) else action)
        if len(states) % 100 == 0:
            logging.debug(f"Collected {len(states)} states/actions")

    states = np.array(states)
    actions = np.array(actions)

    # State Trajectory Plot
    plt.figure()
    plt.plot(states[:, 0], label="Soil Moisture")
    plt.plot(states[:, 1], label="Temperature")
    plt.plot(states[:, 2], label="Nutrient Level")
    plt.plot(states[:, 3], label="Crop Health")
    plt.plot(states[:, 4], label="Season")
    plt.title("State Trajectory Over Time")
    plt.xlabel("Steps")
    plt.ylabel("State Value")
    plt.legend()
    plt.savefig("state_trajectory.png")
    plt.close()
    logging.info("Saved state_trajectory.png")

    # Action Distribution Histogram
    plt.figure()
    plt.hist(actions[:, 0], bins=20, alpha=0.5, label="Irrigation")
    plt.hist(actions[:, 1], bins=20, alpha=0.5, label="Fertilization")
    plt.hist(actions[:, 2], bins=20, alpha=0.5, label="Crop Choice")
    plt.title("Action Distribution")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("action_distribution.png")
    plt.close()
    logging.info("Saved action_distribution.png")

    logging.info("Main execution completed")


if __name__ == "__main__":
    main()
