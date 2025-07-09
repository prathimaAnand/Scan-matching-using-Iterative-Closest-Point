# Scan matching using Iterative Closest Point - ICP Point Cloud Registration in Python

This repository implements the **Iterative Closest Point (ICP)** algorithm from scratch using NumPy and Matplotlib. The goal is to estimate the optimal **rigid transformation (rotation and translation)** between two 3D point clouds.

## Features

- Point correspondence estimation with a distance threshold
- SVD-based optimal rigid registration (rotation + translation)
- RMSE computation for evaluation
- 3D visualization of aligned point clouds using `matplotlib`

## Input Files

Ensure the following files are in the same directory:

- `pclX.txt`: Source point cloud (Nx3 format)
- `pclY.txt`: Target point cloud (Nx3 format)

These should be plain text files with 3D points separated by spaces.

## How to Run

python icp_registration.py

## What it does
Load and parse the point clouds.

Run the ICP algorithm to align the source to the target.

Display the aligned clouds in 3D.

Output the final rotation matrix, translation vector, and RMSE.

## Output

Rotational Matrix (R): Optimal 3x3 rotation aligning source to target.

Translation Matrix (t): Optimal 3x1 translation vector.

RMSE: Root Mean Squared Error for matched correspondences.

## Understanding data

Target point cloud (pclY.txt) is shown in blue.

Transformed source point cloud (pclX.txt after registration) is shown in yellow.

## Requirements

Install the following Python packages:

```bash
pip install numpy matplotlib
