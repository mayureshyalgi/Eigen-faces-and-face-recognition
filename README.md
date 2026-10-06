# Face Recognition using Eigenfaces (PCA)

Mini project for **UE25MA242A – Mathematical Foundation for AI & Data Science**, PES University.
**Problem Statement 11:** Projections, eigenvectors, Principal Component Analysis, and face recognition algorithms.

## Overview

Face photos are turned into a data matrix, and linear algebra is used to find the main patterns of variation ("eigenfaces"). Each face is compressed onto those patterns and recognised by comparing projections. A least squares classifier and a small CNN are included for comparison.

## Linear algebra workflow

| Step | Concept | Outcome |
|---|---|---|
| 1 | Matrix representation | 64×64 photos flattened into a 320 × 4096 data matrix |
| 2 | Mean and centering | Average face and centered matrix |
| 3 | Rank | Faces span a small subspace (rank ≤ 320) |
| 4 | Covariance, eigenvectors, diagonalization | Symmetric covariance matrix diagonalized as C = PDPᵀ |
| 5 | Eigenfaces (PCA) | Top 50 eigenvectors shown as face images |
| 6 | Orthogonal projection | Each face compressed from 4096 to 50 numbers and reconstructed |
| 7 | Distance (norm) | Nearest-neighbour recognition in eigenface space |
| 8 | Least squares | Best-fit linear classifier on the projections |
| 9 | CNN (comparison) | Accuracy compared against the linear algebra methods |

## Dataset

Olivetti faces (40 people, 10 photos each, 64×64 grayscale), loaded automatically through `sklearn.datasets`.

## Files

- `Face_Recognition_Eigenfaces.ipynb` – the full project notebook
- `Eigenfaces_Math_Concepts.pptx` – presentation explaining the math concepts

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Face_Recognition_Eigenfaces.ipynb
```

Run all cells in order. The dataset downloads on first run.
