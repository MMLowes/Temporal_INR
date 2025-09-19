import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from networks.encodings import FourierFeatureEncoding, HarmonicEncoding

class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    n_encoding_features -- (int) number of Fourier features to generate
    skip_connection -- (boolean) use skip connections if True
    """
    def __init__(self, layers, weight_init=True, omega=30, encoding_type="harmonic", n_encoding_features=0, skip_connection=False, include_original_coords=False):
        """Initialize the network."""
        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.n_encoding_features = n_encoding_features
        layers = layers.copy()
        self.spatial_dims = 3
        self.temporal_dims = 4

        # Initialize Fourier features if specified
        if self.n_encoding_features > 0:
            if encoding_type == "harmonic":
                self.encoding_features = HarmonicEncoding(n_functions=n_encoding_features, include_original_coords=include_original_coords)
                layers[0] = 2*self.n_encoding_features*self.spatial_dims + self.temporal_dims + self.spatial_dims*include_original_coords  # Update input layer size
            elif encoding_type == "fourier":
                self.encoding_features = FourierFeatureEncoding(self.spatial_dims, self.n_encoding_features, include_original_coords=include_original_coords)
                layers[0] = 2*self.n_encoding_features+ self.temporal_dims + self.spatial_dims*include_original_coords  # Update input layer size
            else:
                print(f"Encoding type {encoding_type} not recognized, no encoding added.")
        
        if skip_connection:
            self.skip_layer = self.n_layers // 2
            # layers[self.skip_layer+1] += layers[0]

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            if i == self.skip_layer+1:
                self.layers.append(nn.Linear(layers[i] + layers[0], layers[i + 1]))
            else:
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Weight Initialization
            if weight_init:
                with torch.no_grad():
                    sample_size = 1/layers[i] if i == 0 else 1e-3 if i == self.n_layers-1 else np.sqrt(6/layers[i])/self.omega
                
                    self.layers[-1].weight.uniform_(-sample_size, sample_size)

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""
        if self.n_encoding_features > 0:
            x1 = self.encoding_features(x[:,:self.spatial_dims])
            x = torch.cat([x1, x[:,self.spatial_dims:]], dim=-1)
        x0 = x.clone()
        # Perform sine activation on all layers except for the last one
        for i, layer in enumerate(self.layers[:-1]):
            if i == self.skip_layer:
                x = torch.cat([torch.sin(self.omega * layer(x)), x0], dim=-1)
            else:
                x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)


class MLP(nn.Module):
    def __init__(self, layers):
        """Initialize the network."""

        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)
