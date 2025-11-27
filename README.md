# Loss Landscape Geometry & Optimization Dynamics in Neural Networks

**Author:** Himasish Ghosal  
**Roll No:** MA24M010  
**Context:** FourKites  Assignment

## 📌 Project Overview
This repository contains a rigorous framework for analyzing the geometry of neural network loss landscapes and their relationship to optimization dynamics. Using **PyTorch**, this project implements advanced visualization techniques to probe the "terrain" a neural network navigates during training (MNIST classification).

The goal is to empirically validate connections between **geometric properties** (flat vs. sharp minima) and **model behavior** (generalization and trainability).

## 🚀 Key Features & Methodologies

### 1. Rigorous Landscape Visualization
Standard Euclidean plotting fails to account for the scale-invariance of neural networks. This project implements **Filter-Wise Normalization** (based on *Li et al., 2018*) to generate accurate 1D and 2D contour plots of the loss surface.
- **1D Linear Interpolation:** Visualizing loss along specific direction vectors.
- **2D Surface Contours:** Projecting the high-dimensional parameter space onto two random, filter-normalized axes.

### 2. Curvature Analysis (Hessian Spectrum)
To measure the "sharpness" of the converged minima (which correlates with generalization capability):
- Implemented **Hessian-Vector Products (HvP)** and **Power Iteration**.
- Efficiently estimates the **Top Eigenvalue ($\lambda_{max}$)** of the Hessian matrix without computing the full $N \times N$ matrix.

### 3. Optimization Trajectory Analysis (PCA)
Instead of random directions, this module analyzes the actual path taken by Stochastic Gradient Descent (SGD).
- **Snapshotting:** Captures model weights at periodic training steps.
- **PCA Projection:** Uses Principal Component Analysis to find the dominant directions of variance in the optimization trajectory and visualizes the loss surface along these specific dimensions.

### 4. Mode Connectivity
Investigates the topology between two independently trained solutions.
- Trains two separate models starting from different random seeds.
- Performs linear interpolation to analyze the energy barrier (loss spikes) between two distinct local minima.

---

## 🛠️ Technical Stack
* **Core:** Python 3.x
* **Deep Learning:** PyTorch, Torchvision
* **Computation:** NumPy
* **Visualization:** Matplotlib (3D plotting)

## 📂 Code Structure

The core logic is contained within `FourKites_Assignment.ipynb`:

1.  **`SimpleCNN`**: A standard convolutional architecture for MNIST.
2.  **`normalize_direction`**: The critical utility for creating scale-invariant random directions.
3.  **`hessian_top_eigenvalue`**: An implementation of the Lanczos/Power iteration algorithm for curvature estimation.
4.  **`train_with_snapshots`**: Custom training loop that caches weight states for trajectory analysis.
5.  **`pca_directions`**: Dimensionality reduction for visualizing the optimization path.
6.  **`plot_mode_connectivity`**: Routines for analyzing the barrier between two solutions.

## 📊 Results & Observations

### 1. Loss Surface Topography
The generated 3D surface plots reveal that the model converges to a basin. Using filter-normalization ensures that the "width" of the basin is visually representative of the solution's robustness to weight perturbations.

### 2. Trajectory PCA
By projecting the loss landscape onto the principal components of the SGD trajectory, we observe that the optimization process largely happens within a low-dimensional subspace, moving from high-loss regions into a wide valley.

### 3. Curvature
The estimated top Hessian eigenvalue provides a scalar metric for sharpness. A lower $\lambda_{max}$ typically indicates a flatter minimum, suggesting better generalization to unseen test data.

## 💻 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy matplotlib
    ```

2.  **Run the Notebook:**
    Open `FourKites_Assignment.ipynb` in Jupyter Lab, VS Code, or Google Colab.

3.  **Execution Flow:**
    - Run cells sequentially to train the model.
    - The notebook will automatically download the MNIST dataset.
    - Visualization cells will render 3D plots and Trajectory graphs inline.

## 📚 References
* Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). *Visualizing the Loss Landscape of Neural Nets*. NeurIPS.
* Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., & Wilson, A. G. (2018). *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*. NeurIPS.
* Nikita Gabdullin1, *Investigating generalization capabilities of neural networks by means of
loss landscapes and Hessian analysis (2025)*.
