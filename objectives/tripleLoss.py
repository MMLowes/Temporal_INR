import torch
import torch.nn as nn


class TriplePointLoss(nn.Module):
    def __init__(self):
        super(TriplePointLoss, self).__init__()
        self.loss_function = lambda x, y: torch.mean(torch.linalg.vector_norm(x - y, dim=1))

    def forward(self, x, y, z):
        loss_xy = self.loss_function(x, y)
        loss_xz = self.loss_function(x, z)
        loss_yz = self.loss_function(y, z)
        return (loss_xy + loss_xz + loss_yz) / 3
