# SAR-Image-Processing

This project implements a comprehensive pipeline for Synthetic Aperture Radar (SAR) image processing, focusing on performance comparisons between sequential (CPU), OpenMP (multi-threaded), and CUDA (GPU-accelerated) implementations. It includes detailed profiling, execution time analysis, and visualization tools to assess the efficiency of each approach.

## 📁 Repository Structure

```
SAR-Image-Processing/
├── images/
│   └── sample1.jpg
├── main.cpp
├── openmp_implementation.cpp
├── pixel_imp.cpp
├── sample_testing.cpp
├── compare_execution_times.py
├── img_size_vs_exec_time.py
├── plot_graph.py
├── cuda_execution_times.csv
├── openmp_times.csv
├── pixel_implementation.csv
├── image_size_vs_time.csv
├── execution_time_comparison.png
├── img_size_vs_seq_exec_time.png
├── gprof_report.txt
├── gprof.png
├── LICENSE
└── README.md
```

## 🚀 Features

* **Image Processing Pipeline**: Includes grayscale conversion, rotation, flipping, HSV conversion, brightness adjustment, clipping, histogram equalization, Wiener filtering, and Gaussian filtering.
* **Multiple Implementations**:

  * **Sequential (CPU)**: Baseline implementation in `main.cpp`.
  * **OpenMP (Multi-threaded CPU)**: Parallelized version in `openmp_implementation.cpp`.
  * **CUDA (GPU-accelerated)**: High-performance version in `pixel_imp.cpp`.
* **Performance Analysis**:

  * Execution time measurements for each function.
  * Profiling reports using `gprof`.
  * Visualization scripts to compare performance across implementations.

## 🛠️ Setup & Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SagarMaddela/SAR-Image-Processing.git
   cd SAR-Image-Processing
   ```

2. **Install Dependencies**:

   * Ensure you have OpenCV installed for image processing.
   * For CUDA implementation, ensure CUDA toolkit is installed.
   * Python 3 with `matplotlib` and `pandas` for plotting scripts.

3. **Build the Project**:

   * **Sequential Version**:

     ```bash
     g++ main.cpp -o main -fopenmp `pkg-config --cflags --libs opencv4`
     ```
   * **OpenMP Version**:

     ```bash
     g++ openmp_implementation.cpp -o openmp -fopenmp `pkg-config --cflags --libs opencv4`
     ```
   * **CUDA Version**:

     ```bash
     nvcc pixel_imp.cpp -o cuda `pkg-config --cflags --libs opencv4`
     ```

## 🧪 Usage

1. **Run Sequential Implementation**:

   ```bash
   ./main
   ```

2. **Run OpenMP Implementation**:

   ```bash
   ./openmp
   ```

3. **Run CUDA Implementation**:

   ```bash
   ./cuda
   ```

4. **Generate Performance Plots**:

   ```bash
   python compare_execution_times.py
   python img_size_vs_exec_time.py
   python plot_graph.py
   ```

## 📊 Results & Analysis

* **Execution Time Comparison**:

  * `execution_time_comparison.png`: Visual comparison of execution times across implementations.
  * `img_size_vs_seq_exec_time.png`: Execution time vs. image size for sequential implementation.

* **Profiling Reports**:

  * `gprof_report.txt`: Detailed profiling report generated using `gprof`.
  * `gprof.png`: Visual representation of profiling data.

* **CSV Data**:

  * `cuda_execution_times.csv`, `openmp_times.csv`, `pixel_implementation.csv`, `image_size_vs_time.csv`: Raw execution time data for further analysis.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information and updates, visit the [GitHub Repository](https://github.com/SagarMaddela/SAR-Image-Processing).
