# Physics-Informed CNN Autoencoder for Wind Turbine Fault Detection

An unsupervised Deep Learning approach combining 1D-Convolutional Autoencoders (CNN-AE) with a custom **Physics-Informed Kinematic Penalty** to detect mechanical anomalies in wind turbine vibration data.

## 📌 Project Overview
Traditional data-driven anomaly detection often fails to capture the mechanical realities of dynamic systems. In this project, a baseline CNN Autoencoder was found to be insufficient for certain complex fault signatures. To address this, a **Physics-Informed Score** was integrated into the evaluation pipeline. 

By observing that specific faults exhibit unnatural signal stationarity or linearity, the model incorporates the first and second derivatives of the signal (velocity and acceleration) to penalize physically improbable states.

## 🚀 Key Features
* **1D-CNN Autoencoder:** Built from scratch in PyTorch to reconstruct healthy structural signals.
* **Physics-Informed Penalty:** A custom evaluation metric that calculates physical complexities based on signal derivatives, significantly boosting fault isolation accuracy.
* **Multi-Turbine Scalability:** Successfully tested and validated on simulated wind farm datasets (differentiating healthy state from localized mechanical failures).

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (`torch.nn`)
* **Signal Processing:** SciPy, NumPy (FFT, derivative estimations)
* **Data Visualization:** Matplotlib, Seaborn (Reconstruction Loss Distributions, Heatmaps)

## 🧠 Methodology
1. **Signal Preprocessing:** Z-score normalization and artifact removal from raw vibration data.
2. **Model Training:** The CNN-AE is trained exclusively on "Healthy" data to learn the baseline structural manifold using Mean Squared Error (MSE) loss.
3. **Anomaly Scoring:** During inference, the reconstruction error is fused with the Physics-Informed Kinematic Score. 
4. **Fault Isolation:** High composite anomaly scores trigger fault flags (e.g., successful identification of Turbine 5 and Turbine 7 localized faults).

## 💻 Code Structure
* `model.py`: PyTorch implementation of the 1D-Convolutional Autoencoder.
* `physics_score.py`: Functions computing the kinematic complexities and custom penalty logic.
* `train_evaluate.ipynb`: Full pipeline from data loading to fault detection heatmaps.
