import torch.nn as nn
import torch.nn.functional as F


class Multi_linear_layer(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size, activation=None):
        super(Multi_linear_layer, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, n_layers - 1):
            self.linears.append(nn.Linear(hidden_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))
        self.activation = getattr(F, activation)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        linear = self.linears[-1]
        x = linear(x)
        return x