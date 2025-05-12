import pandas as pd
import matplotlib.pyplot as plt

# Load all CSV files
pixel_df = pd.read_csv("pixel_implementation.csv", sep=None, engine='python')
openmp_df = pd.read_csv("openmp_implementation.csv", sep=None, engine='python')
cuda_df = pd.read_csv("cuda_execution_times.csv", sep=None, engine='python')

# Rename columns for consistency
pixel_df.columns = ['Operation', 'Pixel Time (ms)']
openmp_df.columns = ['Operation', 'OpenMP Time (ms)']
cuda_df.columns = ['Operation', 'CUDA Time (ms)']

# Merge all DataFrames on 'Operation'
merged_df = pixel_df.merge(openmp_df, on='Operation').merge(cuda_df, on='Operation')

# Plotting
plt.figure(figsize=(14, 8))
bar_width = 0.25
x = range(len(merged_df))

# Bar positions
r1 = [i - bar_width for i in x]
r2 = x
r3 = [i + bar_width for i in x]

# Bar plots
plt.bar(r1, merged_df['Pixel Time (ms)'], width=bar_width, label='Pixel')
plt.bar(r2, merged_df['OpenMP Time (ms)'], width=bar_width, label='OpenMP')
plt.bar(r3, merged_df['CUDA Time (ms)'], width=bar_width, label='CUDA')

# Labels and title
plt.xlabel('Operation')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time Comparison of Image Processing Operations')
plt.xticks(x, merged_df['Operation'], rotation=45)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig("execution_time_comparison.png")
print("Bar graph saved as 'execution_time_comparison.png'")

