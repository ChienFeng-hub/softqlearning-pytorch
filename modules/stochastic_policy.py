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
        n_action_samples=1
    ):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs[:, None, :].repeat(1, n_action_samples, 1)
        noise = torch.randn((obs.shape[0], n_action_samples, self.action_sizes))
        input = torch.cat([obs, noise], dim=-1).reshape(-1, self.state_sizes + self.action_sizes)
        act = torch.tanh(self.net(input))
        act = act.reshape(obs.shape[0], n_action_samples, self.action_sizes)
        return act
