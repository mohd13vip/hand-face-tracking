import pandas as pd

csv_path = r"task2_tracking\results\metrics\tracking_metrics.csv"
df = pd.read_csv(csv_path, index_col=0)

total_frames = df["num_frames"].sum()

overall_mota = (df["mota"] * df["num_frames"]).sum() / total_frames
overall_motp = (df["motp"] * df["num_frames"]).sum() / total_frames
overall_idf1 = (df["idf1"] * df["num_frames"]).sum() / total_frames
overall_switches = int(df["num_switches"].sum())

print("=== OVERALL (weighted by num_frames) ===")
print("frames:", int(total_frames))
print("mota:", overall_mota)
print("motp:", overall_motp)
print("idf1:", overall_idf1)
print("num_switches:", overall_switches)
