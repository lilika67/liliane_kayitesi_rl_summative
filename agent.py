from implementation.rendering import RwandaFarmEnvVisualizer
from environment.custom_env import RwandaFarmEnv
from stable_baselines3 import PPO
import pygame

pygame.init()
env = RwandaFarmEnv()
model = PPO.load(
    "models/pg/ppo_farm"
)
visualizer = RwandaFarmEnvVisualizer(env)
for episode in range(3):
    visualizer.render(
        f"episode_{episode}_trained_farm.mp4", is_trained=True, model=model
    )
visualizer.close()
