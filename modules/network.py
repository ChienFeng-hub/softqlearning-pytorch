from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
from torch import nn

ModuleType = Type[nn.Module]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU(),
    ) -> None:
        super().__init__()
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, activ in zip(
            hidden_sizes[:-1], hidden_sizes[1:], activation_list):
            model.append(nn.Linear(in_dim, out_dim))
            model.append(activ)
        if output_dim > 0:
            model.append(nn.Linear(hidden_sizes[-1], output_dim))
            self.output_dim = output_dim
        else:
            self.output_dim = hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.model(x)
