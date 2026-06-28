import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import CNN1DAutoencoder
from physics_utils import compute_anomaly_score

def load_and_preprocess(filepath, key):
    """ Loads .mat files and scales data. """
    mat = scipy.io.loadmat(filepath)
    data = mat[key]
    data = np.transpose(data, (0, 2, 1))
    
    # Reshape and scale
    data_2d = data.reshape(data.shape[0], -1)
    scaled_2d = StandardScaler().fit_transform(data_2d)
    
    return scaled_2d.reshape(data.shape)

if __name__ == "__main__":
    print("Loading datasets...")
    # Load data (adjust paths as needed based on your directory structure)
    d1_scaled = load_and_preprocess('../data/D1.mat', 'D1')
    d2_scaled = load_and_preprocess('../data/D2.mat', 'D2')
    
    print("Datasets loaded successfully. Ready for training & inference pipeline.")
    # Here goes the rest of your DataLoader, Training loop, and Inference logic.
    # Focus on using the CNN1DAutoencoder and compute_anomaly_score during evaluation!
