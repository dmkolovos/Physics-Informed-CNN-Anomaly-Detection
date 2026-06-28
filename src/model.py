import torch
import torch.nn as nn

class CNN1DAutoencoder(nn.Module):
    """
    1D-Convolutional Autoencoder for reconstructing healthy vibration signals.
    """
    def __init__(self, sequence_length, num_features):
        super(CNN1DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=num_features, kernel_size=2, stride=2),
            nn.Sigmoid() # Assuming normalized inputs
        )

    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
