import torch

import torch.nn as nn

class VelocityLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(VelocityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, velocities):
        diff = torch.abs(velocities[:, 1:] - velocities[:, :-1])
        loss = torch.mean(torch.exp(self.alpha * diff) - 1)
        return loss