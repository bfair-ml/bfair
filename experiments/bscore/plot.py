import matplotlib.pyplot as plt
import json

# File path to the .jsonl file
file_path = "runs-victoria/v10 bscore-continuous (updated plurals)/leading/.scores-all-leading.jsonl"

# Initialize lists to store data
model_labels = []  # New list to store model labels
count_disparity_means = []
count_disparity_stds = []  # Rename to store standard deviations
log_score_means = []
log_score_stds = []  # Rename to store standard deviations

# Read and parse the .jsonl file
with open(file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        for model, metrics in data.items():
            model_labels.append(model)  # Add the model key as a label
            count_disparity_means.append(metrics["count_disparity"]["mean"])
            count_disparity_stds.append(
                metrics["count_disparity"]["standard_deviation"]
            )  # Store standard deviation
            log_score_means.append(metrics["log_score"]["mean"])
            log_score_stds.append(
                metrics["log_score"]["standard_deviation"]
            )  # Store standard deviation

# Plotting
models = model_labels  # Use model_labels instead of generic "Model {i+1}"

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot count_disparity
axes[0].bar(
    models,
    count_disparity_means,
    yerr=count_disparity_stds,
    capsize=5,
    color="skyblue",
    label="Mean ± Std Dev",
)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)  # Highlight y=0
axes[0].set_title("Count Disparity")
axes[0].set_ylabel("Mean")
axes[0].legend()

# Plot log_score
axes[1].bar(
    models,
    log_score_means,
    yerr=log_score_stds,
    capsize=5,
    color="salmon",
    label="Mean ± Std Dev",
)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)  # Highlight y=0
axes[1].set_title("Log Score")
axes[1].set_ylabel("Mean")
axes[1].legend()

# Add common labels and adjust layout
plt.xlabel("Models")
plt.tight_layout()

# Save the plot to disk
output_path = "output_plot.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
