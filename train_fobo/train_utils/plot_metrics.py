import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FuncFormatter

# Load data from JSON file
with open("results_ppo.json") as file:
    data = json.load(file)

# Create a DataFrame
df = pd.DataFrame(data)

# Convert 'Checkpoint' to integers for proper sorting
df["Checkpoint"] = df["Checkpoint"].astype(int)

# Calculate the mean of 'Reward_list' and 'Steps' for each checkpoint
df["Mean_Reward"] = df["Reward_list"].apply(lambda x: sum(x) / len(x))
df["Mean_Steps"] = df["Steps"].apply(lambda x: sum(x) / len(x))

# Sort the DataFrame by 'Checkpoint'
df = df.sort_values(by="Checkpoint")

print("Checkpoints and their corresponding Mean Rewards and Mean Steps:")
print(df[["Checkpoint", "Mean_Reward", "Mean_Steps"]])


# Formatter function to change x-axis labels
def scientific_formatter(x, pos):
    return f"{int(x / 10000)}"


# Plotting
fig, ax1 = plt.subplots(constrained_layout=True)

color = "tab:red"
ax1.set_xlabel("Checkpoint (× 10^4)")
ax1.set_ylabel("Mean Reward", color=color)
ax1.plot(df["Checkpoint"], df["Mean_Reward"], color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = "tab:blue"
ax2.set_ylabel("Mean Steps", color=color)
ax2.plot(df["Checkpoint"], df["Mean_Steps"], color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Mean Reward and Mean Steps by Checkpoint with PPO model")
fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Save the plot to a file instead of showing it
plt.savefig("plot.png")
