import gymnasium as gym
from environment.custom_env import RwandaFarmEnv
import pygame
import numpy as np
import imageio
import os
import logging
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

pygame.init()
screen = pygame.display.set_mode((800, 576))
pygame.display.set_caption("Rwanda Farm Simulation - Interactive View")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Load tractor image as agent sprite
tractor_img = pygame.image.load(os.path.join("assets", "tractor.png")).convert_alpha()
tractor_img = pygame.transform.scale(tractor_img, (40, 40))  # Resize to 40x40 pixels


class RwandaFarmEnvVisualizer:
    def __init__(self, env):
        self.env = env
        self.frames = []
        self.max_frames = 90
        self.screen = screen
        self.paused = False
        self.crops = [
            (100 + i * 100, 200 + j * 100) for i in range(3) for j in range(2)
        ]  # 3x2 crop grid
        self.water_sources = [(50, 50), (700, 50)]  # Two water sources
        self.agent_pos = [400, 288]  # Start center of screen as instance variable

    def render(self, filename=None, is_trained=False, model=None):
        state, _ = self.env.reset()
        done = False
        frame_count = 0
        logging.info(f"Render started with initial state: {state}")
        

        while not done and frame_count < self.max_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.info("Render aborted by user")
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                        logging.info(
                            f"Pause toggled: {'Paused' if self.paused else 'Resumed'}"
                        )
                    if event.key == pygame.K_r:
                        state, _ = self.env.reset()
                        frame_count = 0
                        done = False
                        self.agent_pos = [400, 288]  # Reset agent position
                        logging.info("Reset simulation")

            if self.paused:
                continue

            action = (
                self.env.action_space.sample()
                if not is_trained
                else model.predict(state, deterministic=True)[0]
            )
            if isinstance(action, tuple):
                action = action[0]
            state, reward, done, _, info = self.env.step(action)
            logging.debug(
                f"Frame {frame_count}: Action={action}, Crop Health={state[3]}, Reward={reward}"
            )

            # Draw farm layout
            self.screen.fill((34, 139, 34))  # Darker green background
            for crop_x, crop_y in self.crops:
                pygame.draw.rect(
                    self.screen, (144, 238, 144), (crop_x, crop_y, 50, 50)
                )  # Light green crops
            for water_x, water_y in self.water_sources:
                pygame.draw.circle(
                    self.screen, (135, 206, 235), (water_x, water_y), 20
                )  # Sky blue water sources

            # Move agent randomly
            self.agent_pos[0] += random.randint(-20, 20)
            self.agent_pos[1] += random.randint(-20, 20)
            self.agent_pos = [
                max(20, min(780, x)) for x in self.agent_pos
            ]  # Keep within screen
            self.screen.blit(
                tractor_img, (self.agent_pos[0] - 20, self.agent_pos[1] - 20)
            )  # Center tractor

            # Simulate action effects
            if action[0] > 0:  # Irrigation
                pygame.draw.line(
                    self.screen,
                    (135, 206, 235),
                    self.agent_pos,
                    (self.agent_pos[0], 0),
                    5,
                )  # Water spray
            if action[1] > 0:  # Fertilization
                pygame.draw.circle(
                    self.screen, (255, 215, 0), self.agent_pos, 20
                )  # Yellow fertilizer spread

            # Add interactive labels
            health_text = font.render(
                f"Crop Health: {state[3]:.2f}", True, (255, 255, 255)
            )
            moisture_text = font.render(
                f"Soil Moisture: {state[0]:.2f}", True, (255, 255, 255)
            )
            self.screen.blit(health_text, (150, 350))
            self.screen.blit(moisture_text, (300, 380))

            pygame.display.flip()

            if filename:
                pygame.time.wait(100)
                frame = pygame.surfarray.array3d(self.screen)
                if frame.shape == (800, 576, 3):
                    frame = np.swapaxes(frame, 0, 1)
                    self.frames.append(frame)
                    logging.debug(f"Frame {frame_count} captured: shape={frame.shape}")
                else:
                    logging.warning(f"Unexpected frame shape: {frame.shape}")

            clock.tick(10)
            frame_count += 1

        if filename and self.frames:
            try:
                if "gif" in filename.lower():
                    imageio.mimsave(filename, self.frames, fps=10)
                    logging.info(f"Saved {filename}")
                elif "mp4" in filename.lower():
                    imageio.mimwrite(filename, self.frames, fps=10, quality=8)
                    logging.info(f"Saved {filename}")
            except Exception as e:
                logging.error(f"Failed to save {filename}: {e}")
        else:
            logging.error(f"No frames captured for {filename}")
        self.frames = []

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = RwandaFarmEnv()
    visualizer = RwandaFarmEnvVisualizer(env)
    visualizer.render("random_farm.gif")
    # For trained visualization, uncomment and provide model:
    # visualizer.render("trained_farm.mp4", is_trained=True, model=your_trained_model)
