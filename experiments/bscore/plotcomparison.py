import matplotlib.pyplot as plt
import json

# File paths to the .jsonl files
leading_file_path = "runs-victoria/v8 bscore-continuous (morph)/leading/.scores-all-leading.jsonl"
non_leading_file_path = "runs-victoria/v8 bscore-continuous (morph)/no-leading/.scores-all-independent.jsonl"

# Initialize lists to store data for both files
model_labels_leading = []
count_disparity_means_leading = []
count_disparity_std_leading = []
count_disparity_errors_leading = []
log_score_means_leading = []
log_score_std_leading = []
log_score_errors_leading = []

model_labels_non_leading = []
count_disparity_means_non_leading = []
count_disparity_std_non_leading = []
count_disparity_errors_non_leading = []
log_score_means_non_leading = []
log_score_std_non_leading = []
log_score_errors_non_leading = []

# Function to parse a .jsonl file and populate lists
def parse_jsonl(file_path, count_means, count_std, count_errors, log_means, log_std, log_errors, labels):
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            for model, metrics in data.items():
                model_name = model.split("-")[0]
                labels.append(model_name)
                count_means.append(metrics["count_disparity"]["mean"])
                count_std.append(metrics["count_disparity"]["standard_deviation"])
                count_errors.append(metrics["count_disparity"]["confidence_margin"])
                log_means.append(metrics["log_score"]["mean"])
                log_std.append(metrics["log_score"]["standard_deviation"])
                log_errors.append(metrics["log_score"]["confidence_margin"])

# Parse both files with standard deviation
parse_jsonl(leading_file_path, count_disparity_means_leading, count_disparity_std_leading, count_disparity_errors_leading, log_score_means_leading, log_score_std_leading, log_score_errors_leading, model_labels_leading)
parse_jsonl(non_leading_file_path, count_disparity_means_non_leading, count_disparity_std_non_leading, count_disparity_errors_non_leading, log_score_means_non_leading, log_score_std_non_leading, log_score_errors_non_leading, model_labels_non_leading)
assert model_labels_leading == model_labels_non_leading, "Model labels do not match between leading and non-leading files."

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Offset for non-overlapping bars
bar_width = 0.4
x = range(len(model_labels_leading))
x_leading = [i - bar_width / 2 for i in x]
x_non_leading = [i + bar_width / 2 for i in x]

# Plot count_disparity
axes[0].bar(x_leading, count_disparity_means_leading, color='skyblue', width=bar_width, label='Leading (Mean)')
axes[0].errorbar(x_leading, count_disparity_means_leading, yerr=count_disparity_std_leading, fmt='none', ecolor='blue', capsize=5, label='Leading (±SD)')
axes[0].errorbar(x_leading, count_disparity_means_leading, yerr=count_disparity_errors_leading, fmt='s', mew=4, mfc="darkblue", mec="darkblue", ecolor='darkblue', capsize=5, label='Leading (±CI)')
axes[0].bar(x_non_leading, count_disparity_means_non_leading, color='orange', width=bar_width, alpha=0.7, label='Non-Leading (Mean)')
axes[0].errorbar(x_non_leading, count_disparity_means_non_leading, yerr=count_disparity_std_non_leading, fmt='none', ecolor='orange', capsize=5, label='Non-Leading (±SD)')
axes[0].errorbar(x_non_leading, count_disparity_means_non_leading, yerr=count_disparity_errors_non_leading, fmt='s', mew=4, mfc="darkorange", mec="darkorange", ecolor='darkorange', capsize=5, label='Non-Leading (±CI)')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_title("Count Disparity")
axes[0].set_ylabel("Mean")
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_labels_leading, rotation=45, ha='right')
axes[0].legend()

# Plot log_score
axes[1].bar(x_leading, log_score_means_leading, color='salmon', width=bar_width, label='Leading (Mean)')
axes[1].errorbar(x_leading, log_score_means_leading, yerr=log_score_std_leading, fmt='none', ecolor='red', capsize=5, label='Leading (±SD)')
axes[1].errorbar(x_leading, log_score_means_leading, yerr=log_score_errors_leading, fmt='s', mew=4, mfc="darkred", mec="darkred", ecolor='darkred', capsize=5, label='Leading (±CI)')
axes[1].bar(x_non_leading, log_score_means_non_leading, color='green', width=bar_width, alpha=0.7, label='Non-Leading (Mean)')
axes[1].errorbar(x_non_leading, log_score_means_non_leading, yerr=log_score_std_non_leading, fmt='none', ecolor='green', capsize=5, label='Non-Leading (±SD)')
axes[1].errorbar(x_non_leading, log_score_means_non_leading, yerr=log_score_errors_non_leading, fmt='s', mew=4, mfc="darkgreen", mec="darkgreen", ecolor='darkgreen', capsize=5, label='Non-Leading (±CI)')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
axes[1].set_title("Log Score")
axes[1].set_ylabel("Mean")
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_labels_non_leading, rotation=45, ha='right')
axes[1].legend()

# Add common labels and adjust layout
plt.xlabel("Models")
plt.tight_layout()

# Save the plot to disk
output_path = "output_plot_comparison.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")