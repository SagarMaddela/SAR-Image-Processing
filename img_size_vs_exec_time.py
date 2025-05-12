import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("image_size_vs_time.csv")

# Sort data by image size (Total Pixels)
df_sorted = df.sort_values(by="Total Pixels")

# Create line plot
plt.figure(figsize=(10, 6))
plt.plot(
    df_sorted["Total Pixels"], 
    df_sorted["Total Execution Time (ms)"], 
    marker='o', linestyle='-', color='blue'
)
plt.title("Image Size vs Total Execution Time")
plt.xlabel("Image Size (Total Pixels: Width × Height)")
plt.ylabel("Execution Time (ms)")
plt.grid(True)
plt.tight_layout()

# Save the figure to file
plt.savefig('img_size_vs_seq_exec_time.png')
print("Plot saved to img_size_vs_seq_exec_time.png")

