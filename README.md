
# Reinforcement Learning Summative Assignment Report - Crop Yield Optimization

## Overview
This project implements a reinforcement learning (RL) solution to optimize crop yield on a simulated Rwandan farm. The agent manages irrigation, fertilization, and crop choice to maximize crop health while minimizing costs, using a 5D state space environment. The methods employed include Deep Q-Network (DQN), REINFORCE, Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C). The project meets rubric goals through visualizations, a 3-minute video, and hyperparameter tuning.

## Project Structure

liliane_kayitesi_rl_summative/
├── environment/
│   ├── __init__.py
│   └── custom_env.py        # Custom environment definition (RwandaFarmEnv)
├── implementation/
│   └── rendering.py         # Pygame visualization script
├── assets/
│   └── tractor.png          # Agent sprite image
├── models/
│   └── pg/                  # Trained PPO model (ppo_farm)
│   └── dqn/                 # Trained DQN model (dqn_farm)
├── training_log_20250731_full.txt  # Training logs
├── random_farm.gif          # Random agent visualization
├── trained_farm.mp4         # Trained agent visualization
├── DQN_objective_plot.png   # DQN objective function plot
├── PG_entropy_plot.png      # PG policy entropy plot
├── state_trajectory_plot.png # State trajectory plot
├── env/                     # Virtual environment
└── README.md                # This file


## Setup Instructions
1. **Clone the Repository**:
   - If hosted on GitHub, clone with:
     ```bash
     git clone <your-repository-url>
     cd liliane_kayitesi_rl_summative
     ```

2. **Set Up Virtual Environment**:
   - Create and activate a virtual environment:
     ```bash
     python -m venv env
     source env/bin/activate  # On macOS/Linux

     ```

3. **Install Dependencies**:
   - Install required packages:
     ```bash
     pip install gymnasium pygame imageio numpy stable-baselines3 matplotlib
     ```
   - Ensure `assets/tractor.png` is in the `assets` directory.

4. **Verify Environment**:
   - Check the `environment/custom_env.py` file contains the `RwandaFarmEnv` class with the 5D state space and reward function `R = 1500 * crop_health - 30 * (irrigation² + fertilization²) + 150 * crop_choice`.

## Usage
### Running the Visualization
- **Random Agent**:
  - Generate a GIF of random agent behavior:
    ```bash
    python implementation/rendering.py
    ```
  - Output: `random_farm.gif`

- **Trained Agent (PPO)**:
  


## Results
### Performance Metrics
- **Episodes to Stable Performance**:
  - DQN: ~50 episodes, reward ~7650.12.
  - PPO: ~150 episodes, reward peaks at 90,000 then diverges.
  - REINFORCE: No stability, reward ~5000.
  - A2C: No stability, reward ~4000.
- **Cumulative Reward**: Visualized in `reward_plot.png`.
- **Training Stability**:
  - DQN Objective: Stabilizes at 50 episodes (`DQN_objective_plot.png`).
  - PG Entropy: PPO stabilizes at 150 with divergence, REINFORCE/A2C flat (`PG_entropy_plot.png`).
- **Generalization**: PPO (~9000), DQN (~7000), REINFORCE (~5000), A2C (~4000) on unseen states.

### Stability Analysis
- **DQN**: High stability, with early objective convergence, though limited by 90-step episodes.
- **PPO**: Moderate stability, with initial convergence at 150 episodes but post-divergence instability.
- **REINFORCE**: Poor stability, no learning due to variance.
- **A2C**: Poor stability, no convergence due to tuning issues.


