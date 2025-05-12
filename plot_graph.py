import pandas as pd
import matplotlib.pyplot as plt

# Use comma separator
opencv_df = pd.read_csv("openmp_implementation.csv", sep=',', encoding='utf-8-sig')
manual_df = pd.read_csv("pixel_implementation.csv", sep=',', encoding='utf-8-sig')

# Strip column headers
opencv_df.columns = opencv_df.columns.str.strip()
manual_df.columns = manual_df.columns.str.strip()

# Print for debugging
print("OpenMP Columns:", opencv_df.columns.tolist())
print("Manual Columns:", manual_df.columns.tolist())

# Merge
merged_df = pd.merge(opencv_df, manual_df, on="Operation", suffixes=('_OpenCV', '_Manual'))

# Plotting
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(merged_df))

plt.bar(index, merged_df["Execution Time (ms)_OpenCV"], bar_width, label='OpenMP', color='skyblue')
plt.bar([i + bar_width for i in index], merged_df["Execution Time (ms)_Manual"], bar_width, label='Manual', color='salmon')

plt.xlabel("Image Processing Operation")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time Comparison: OpenMP vs Manual Implementation")
plt.xticks([i + bar_width / 2 for i in index], merged_df["Operation"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("execution_time_comparison.png")
