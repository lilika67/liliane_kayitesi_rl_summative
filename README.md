
### Markdown Code Block for README.md
```
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
     env\Scripts\activate     # On Windows
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
  - Visualize the trained PPO model:
    ```bash
    python -c "from implementation.rendering import RwandaFarmEnvVisualizer; from environment.custom_env import RwandaFarmEnv; from stable-baselines3 import PPO; env = RwandaFarmEnv(); model = PPO.load('models/pg/ppo_farm'); visualizer = RwandaFarmEnvVisualizer(env); visualizer.render('trained_farm.mp4', is_trained=True, model=model)"
    ```
  - Output: `trained_farm.mp4`

### Generating Plots
- Run the script to create objective and entropy plots:
  ```bash
  python plot_metrics.py
  ```
  - Outputs: `DQN_objective_plot.png`, `PG_entropy_plot.png`
  - Note: Update `plot_metrics.py` with actual log parsing logic for `training_log_20250731_full.txt`.

### Recording Video
- Use QuickTime Player (macOS):
  1. File > New Screen Recording.
  2. Select the Pygame window (800x576).
  3. Run the trained agent script above.
  4. Record 3 episodes (extend to 3 minutes), save as `agent_simulation.mp4`.
  - Upload to Google Drive and insert the link in the report.

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

## Submission
- **Package Files**:
  ```bash
  tar -czf submission_20250802.tar.gz *.gif *.mp4 *.png models/pg/ models/dqn/ report.pdf
  ```
- **Upload**: Submit with note "Updated on August 02, 2025."

## Acknowledgments
- Built with xAI’s Grok 3 assistance.
- Thanks to course instructors for guidance.

## License
[Insert license, e.g., MIT] - Optional, consult your institution.
```

### Implementation Steps
1. **Create README.md**:
   - Save the content above as `README.md` in `/Users/liliane/Documents/liliane_kayitesi_rl_summative/`.
   - Use a text editor (e.g., VS Code, Nano) and paste the entire code block, ensuring the Markdown syntax is preserved.

2. **Verify Files**:
   - Ensure all listed files (e.g., `random_farm.gif`, `models/pg/ppo_farm`) exist:
     ```bash
     ls -l /Users/liliane/Documents/liliane_kayitesi_rl_summative/{*.gif,*.mp4,*.png,models/pg/,models/dqn/,environment/,assets/,training_log_20250731_full.txt}
     ```

3. **Update Report Reference**:
   - In your Google Docs report, link to `README.md` under "GitHub Repository" once uploaded (e.g., `https://github.com/yourusername/rl-crop-yield/blob/main/README.md`).

4. **Test Rendering**:
   - Confirm the README context works with your project:
     ```bash
     cd /Users/liliane/Documents/liliane_kayitesi_rl_summative/
     source env/bin/activate
     python implementation/rendering.py
     ```

### Verification
- **Check Outputs**:
  ```bash
  ls -l /Users/liliane/Documents/liliane_kayitesi_rl_summative/README.md *.gif *.mp4 *.png models/*
  cat /Users/liliane/Documents/liliane_kayitesi_rl_summative/README.md  # Confirm content
  `

