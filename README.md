#  Reinforcement Learning Summative Assignment Report - Crop Yield Optimization

##  Overview
This project implements a reinforcement learning (RL) solution to optimize crop yield on a simulated Rwandan farm. The agent manages irrigation, fertilization, and crop choice to maximize crop health while minimizing costs, using a 5D state space environment.

Algorithms used:
- Deep Q-Network (DQN)
- REINFORCE
- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)

---

##  Project Structure

```
liliane_kayitesi_rl_summative/
├── environment/
│   ├── __init__.py
│   └── custom_env.py          # Custom environment definition (RwandaFarmEnv)
├── implementation/
│   └── rendering.py           # Pygame visualization script
├── assets/
│   └── tractor.png            # Agent sprite image
├── models/
│   ├── pg/                    # Trained PPO model (ppo_farm)
│   └── dqn/                   # Trained DQN model (dqn_farm)
├── training_log_20250731_full.txt  # Training logs
├── random_farm.gif            # Random agent visualization
├── trained_farm.mp4           # Trained agent visualization
├── DQN_objective_plot.png     # DQN objective function plot
├── PG_entropy_plot.png        # PG policy entropy plot
├── state_trajectory_plot.png  # State trajectory plot
├── env/                       # Virtual environment (excluded from Git)
└── README.md                  # Project overview
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/lilika67/liliane_kayitesi_rl_summative.git
cd liliane_kayitesi_rl_summative
```

### 2. Set Up Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install gymnasium pygame imageio numpy stable-baselines3 matplotlib
```

Make sure `assets/tractor.png` exists in the `assets` folder.

### 4. Verify Custom Environment

Ensure `environment/custom_env.py` contains the class `RwandaFarmEnv` and a reward function:

```python
R = 1500 * crop_health - 30 * (irrigation**2 + fertilization**2) + 150 * crop_choice
```

---

##  Usage

###  Run Visualization

**Random Agent:**

```bash
python implementation/rendering.py
```

- Output: `random_farm.gif`

**Trained PPO Agent:**

To visualize trained PPO behavior (you’ll need to load from the trained model):

```bash
# Replace with correct command if different
python implementation/rendering.py --model models/pg/ppo_farm.zip
```

---

##  Results

###  Performance Metrics

| Algorithm   | Convergence | Reward Range     |
|-------------|-------------|------------------|
| DQN         | ~50 episodes | ~7,650.12        |
| PPO         | ~150 episodes | Peaks ~90,000    |
| REINFORCE   | No convergence | ~5,000         |
| A2C         | No convergence | ~4,000         |

- `reward_plot.png` shows total reward progression
- `DQN_objective_plot.png`: DQN stabilizes early
- `PG_entropy_plot.png`: PPO entropy drops after episode 150
- `state_trajectory_plot.png`: shows state transitions

###  Stability Summary

- **DQN**: High stability, but performance limited by short episode horizon (90 steps).
- **PPO**: Strong early performance; diverges post-episode 150.
- **REINFORCE**: Poor performance due to high variance.
- **A2C**: No significant learning; hyperparameter tuning ineffective.

### 🌍 Generalization to Unseen States

| Method     | Avg. Reward on Unseen States |
|------------|------------------------------|
| PPO        | ~9,000                       |
| DQN        | ~7,000                       |
| REINFORCE  | ~5,000                       |
| A2C        | ~4,000                       |

---

## 🎥 Video Demo

📽️ [Agent in action](https://www.loom.com/share/8d7ef2d24dd643cca7c67396752cb7df?sid=d66e32ed-71a3-4007-a3cc-9e46843519cd)]

---


