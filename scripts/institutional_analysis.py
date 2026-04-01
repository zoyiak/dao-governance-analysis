import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
summary = pd.read_csv("data/dao_summary.csv")
conditions = pd.read_csv("data/institutional_conditions.csv")

# Compute per-DAO averages for Gini and top-10 share
dao_avg = summary.groupby("dao").agg(
    gini_coefficient=("gini_coefficient", "mean"),
    top10_share=("top10_share", "mean"),
).reset_index()

# Merge with institutional conditions
merged = dao_avg.merge(conditions, on="dao")

# Create binary grouping columns
merged["has_identity"] = merged["identity_mechanism"].isin(["Yes", "Partial"])
merged["has_gov_sep"] = merged["gov_fin_separated"].isin(["Yes", "Partial"])

# --- Chart: 2x2 grouped bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors
colors = {
    (True, True): "#2ecc71",
    (True, False): "#f39c12",
    (False, True): "#3498db",
    (False, False): "#e74c3c",
}
labels = {
    (True, True): "Identity + Gov Sep",
    (True, False): "Identity only",
    (False, True): "Gov Sep only",
    (False, False): "Neither",
}

for ax, metric, title in [
    (axes[0], "gini_coefficient", "Avg Gini Coefficient"),
    (axes[1], "top10_share", "Avg Top-10 Concentration"),
]:
    # Sort by metric for readability
    plot_data = merged.sort_values(metric)
    bars = ax.bar(
        plot_data["dao"],
        plot_data[metric],
        color=[
            colors[(id_, gs)]
            for id_, gs in zip(plot_data["has_identity"], plot_data["has_gov_sep"])
        ],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(title)
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Build legend from actually-used groups
used_groups = set(zip(merged["has_identity"], merged["has_gov_sep"]))
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[k], label=labels[k])
    for k in sorted(used_groups, reverse=True)
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "DAO Governance Concentration by Institutional Design",
    fontsize=15, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("outputs/institutional_conditions_chart.png", dpi=200, bbox_inches="tight")
print("Chart saved to outputs/institutional_conditions_chart.png")

# Print merged data for reference
print("\nMerged dataset:")
print(merged[["dao", "gini_coefficient", "top10_share", "identity_mechanism",
              "gov_fin_separated", "has_identity", "has_gov_sep"]].to_string(index=False))
