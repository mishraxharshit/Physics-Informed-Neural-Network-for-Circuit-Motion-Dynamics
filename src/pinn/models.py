"""Small fully-connected network used by every experiment."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, hidden: int = 32, depth: int = 3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]  # tanh: smooth, needed for 2nd derivatives
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx via autograd, keeping the graph so higher derivatives and training work."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
