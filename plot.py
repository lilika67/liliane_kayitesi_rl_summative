import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Simulated data based on your inputs (replace with actual log data)
def load_training_data(log_file):
    data = defaultdict(list)
    with open(log_file, "r") as f:
        for line in f:
            if "DQN Objective" in line:
                episode = int(line.split("Episode ")[1].split(":")[0])
                value = float(line.split("Value: ")[1].split()[0])
                data["dqn_objective"].append((episode, value))
            elif "PG Entropy" in line:
                episode = int(line.split("Episode ")[1].split(":")[0])
                value = float(line.split("Value: ")[1].split()[0])
                data["pg_entropy"].append((episode, value))
    return data


# Load data (replace with your log file path)
log_file = "training_log_20250731.txt"
data = load_training_data(log_file)

# Extract and sort data
dqn_objective = sorted(data["dqn_objective"])
pg_entropy = sorted(data["pg_entropy"])

# Plot DQN Objective Function
plt.figure(figsize=(10, 6))
plt.plot(
    [x[0] for x in dqn_objective],
    [x[1] for x in dqn_objective],
    label="DQN Objective",
    color="blue",
)
plt.axvline(x=50, color="red", linestyle="--", label="Stability at ~50 episodes")
plt.title("DQN Objective Function Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Objective Value")
plt.legend()
plt.grid(True)
plt.savefig("DQN_objective_plot.png")
plt.close()

# Plot PG Policy Entropy
plt.figure(figsize=(10, 6))
for method in ["REINFORCE", "PPO", "A2C"]:
    # Simulate entropy decay (replace with actual data)
    episodes = [
        x[0] for x in pg_entropy if any(m in log_file for m in [method.lower()])
    ]
    values = [x[1] for x in pg_entropy if any(m in log_file for m in [method.lower()])]
    if method == "PPO":
        plt.plot(episodes, values, label=f"{method} Entropy", color="green")
        plt.axvline(
            x=150, color="red", linestyle="--", label="Stability at ~150 episodes"
        )
    elif method == "REINFORCE":
        plt.plot(
            episodes, values, label=f"{method} Entropy", color="orange", linestyle="--"
        )
    elif method == "A2C":
        plt.plot(
            episodes, values, label=f"{method} Entropy", color="purple", linestyle="--"
        )
plt.title("Policy Entropy Over Episodes for PG Methods")
plt.xlabel("Episode")
plt.ylabel("Entropy Value")
plt.legend()
plt.grid(True)
plt.savefig("PG_entropy_plot.png")
plt.close()

print("Plots generated: DQN_objective_plot.png, PG_entropy_plot.png")
