import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
csv_path = os.path.join(HERE, "comparison.csv")
out_dir = os.path.join(HERE, "plots")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# find model column
model_col = None
for c in df.columns:
    if c.lower() in ["model", "name", "variant"]:
        model_col = c
        break
if model_col is None:
    model_col = df.columns[0]

df[model_col] = df[model_col].astype(str)

numeric_cols = [c for c in df.columns if c != model_col and pd.api.types.is_numeric_dtype(df[c])]

def barplot(metric: str):
    plt.figure()
    plt.bar(df[model_col], df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"{metric} by model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

# plot common important metrics first (if present)
priority_names = [
    "mAP50-95", "mAP50_95", "map50_95",
    "mAP50", "map50",
    "precision", "recall",
    "fps", "FPS"
]

done = set()
for want in priority_names:
    for c in df.columns:
        if c.lower() == want.lower() and c in numeric_cols and c not in done:
            barplot(c)
            done.add(c)

# plot remaining numeric columns
for c in numeric_cols:
    if c not in done:
        barplot(c)

print("\nAll plots saved in:", out_dir)
