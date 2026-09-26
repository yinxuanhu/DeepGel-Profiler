# DeepGel-Profiler: High-Throughput Microstructure Analysis for Food Gels

## Overview
This repository contains the Python implementation for the paper "Mung bean protein fibrils reinforce whey protein gels at neutral pH: Effects of fibril maturity, concentration, and microstructure". 

It utilizes Deep Transfer Learning (ResNet-34) to extract high-dimensional morphological features from Confocal Laser Scanning Microscopy (CLSM) images of WPI-MF composite gels. By projecting these features into a latent space using PCA, we quantitatively reveal the kinetic evolution and dosage effects of microfibrillated cellulose on protein networks.

## Methodology
1.  **Feature Extraction**: Uses a pre-trained ResNet-34 model (ImageNet weights) as a fixed feature extractor to capture 512-dimensional texture fingerprints.
2.  **Dimensionality Reduction**: Applies Principal Component Analysis (PCA) to visualize structural clustering and trajectories.
3.  **Visualization**: Decouples the analysis into Kinetic Trajectories (Time Effect) and Dosage Effects (Concentration).

## Key Findings
* **Kinetics**: Thermal induction time is the primary determinant of structural homogeneity, with a clear trajectory from 4h (immature) to 12h (thermodynamic equilibrium).
* **Percolation**: A structural percolation threshold was identified at 0.3% MF, marking the transition to a stable, fibril-reinforced network.

## Usage
1.  Install dependencies: `pip install -r requirements.txt`
2.  Place CLSM images in `data/images/` (Naming format: `12WPI_0.3MF_4h_01.tif`).
3.  Run the analysis: `python src/microstructure_profiling.py`

## Citation
If you use this code, please cite: Mung bean protein fibrils reinforce whey protein gels at neutral pH: Effects of fibril maturity, concentration, and microstructure
