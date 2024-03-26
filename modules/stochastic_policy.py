import torch
from torch import nn
from modules.network import MLP


class StochasticNNPolicy(nn.Module):
    def __init__(
        self, 
        state_sizes,
        action_sizes,
        hidden_sizes,
        device
    ):
        super().__init__()
        self.device = device
        self.net = MLP(
            input_dim=state_sizes+action_sizes,
            output_dim=action_sizes,
            hidden_sizes=hidden_sizes,
        )
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes

    def forward(
        self, 
        obs,
    ):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        noise = torch.randn((obs.shape[0], self.action_sizes), device=self.device)
        input = torch.cat([obs, noise], dim=1)
        act = torch.tanh(self.net(input))
        return act
