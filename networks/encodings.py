import torch
import torch.nn as nn
import numpy as np

class FourierFeatureEncoding(nn.Module):
    """This class generates Fourier encodings for input data.

    Arguments:
    input_dim -- (int) dimension of the input data
    mapping_size -- (int) number of Fourier features to generate
    scale -- (float) scale factor for the Fourier features
    """

    def __init__(self, input_dim, mapping_size, scale=10, include_original_coords=False):
        """Initialize the FourierFeatures module."""
        super(FourierFeatureEncoding, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.include_original_coords = include_original_coords

        # Randomly initialize the weight matrix for Fourier features
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False).cuda()

    def forward(self, x):
        """
        Forward pass to generate Fourier features.
        Arguments:
        x -- (torch.Tensor) input tensor of shape (batch_size, input_dim)
        Returns:
        torch.Tensor -- tensor of shape (batch_size, 2 * mapping_size (+ input_dim)) containing the Fourier features
        """
        x = (x + 1) / 2 # from [-1, 1] to [0, 1]
        x_proj = 2 * np.pi * x @ self.B
        
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x] 
                            if self.include_original_coords else 
                            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


      
class HarmonicEncoding(torch.nn.Module):
    """
    This class generates harmonic encodings for input data.
    """
    def __init__(self, n_functions=6, base_omega_0=1.0, include_original_coords=False):
        
        super().__init__()
                
        frequencies = 2.0 ** torch.arange(n_functions, dtype=torch.float32)
        
        self.register_buffer('frequencies', frequencies*base_omega_0, persistent=True)
        self.include_original_coords = include_original_coords

    def forward(self, x):
        """
        Forward pass to generate harmonic encodings.
        Arguments:
        x -- (torch.Tensor) input tensor of shape (batch_size, input_dim)
        Returns:
        torch.Tensor -- tensor of shape (batch_size, 2 * n_functions (+ input_dim)) containing the harmonic encodings
        """
        
        x1 = (x + 1) / 2 + 1 # from [-1, 1] to [1, 2]
        x_encoded = (x1[..., None] * self.frequencies).reshape(*x.shape[:-1], -1)
                
        return torch.cat((x_encoded.sin(), x_encoded.cos(), x)
                              if self.include_original_coords else 
                              (x_encoded.sin(), x_encoded.cos()), dim=-1)