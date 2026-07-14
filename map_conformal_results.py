import os

import matplotlib.pyplot as plt
import pandas as pd


input_path = "results/conformal_q_hat_by_snow_day.csv"
output_path = "results/conformal_q_hat_by_snow_day.png"

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Missing {input_path}. Run training with conformal.enabled=true first."
    )

conformal = pd.read_csv(input_path).sort_values("snow_day")

print(f"Loaded conformal interval data for {len(conformal)} snow days")
print("Conformal interval half-width statistics:")
print(f"  Mean:   {conformal['conformal_q_hat'].mean():.3f}")
print(f"  Median: {conformal['conformal_q_hat'].median():.3f}")
print(f"  Min:    {conformal['conformal_q_hat'].min():.3f}")
print(f"  Max:    {conformal['conformal_q_hat'].max():.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    conformal["snow_day"],
    conformal["conformal_q_hat"],
    color="tab:blue",
    linewidth=1.8,
)
ax.set_xlabel("Day of snow year")
ax.set_ylabel("Conformal interval half-width")
ax.set_title("Split-conformal half-width by day of snow year")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot to {output_path}")
