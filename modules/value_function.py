import torch
from torch import nn
from modules.network import MLP


class NNQFunction(nn.Module):
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
            output_dim=1,
            hidden_sizes=hidden_sizes,
        )

    def forward(
        self, 
        obs,
        act,
    ):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        input = torch.cat([obs, act], dim=1)
        q = self.net(input)
        return q
