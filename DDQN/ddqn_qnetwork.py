import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc1_units=64,
        fc2_units=64,
        fc3_units=0,
        fc4_units=0,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer (optional)
            fc4_units (int): Number of nodes in fourth hidden layer (optional)
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        if fc3_units != 0:
            self.fc3 = nn.Linear(fc2_units, fc3_units)
        if fc4_units != 0:
            self.fc4 = nn.Linear(fc3_units if fc3_units != 0 else fc2_units, fc4_units)
        self.fc5 = nn.Linear(
            (
                fc4_units
                if fc4_units != 0
                else fc3_units
                if fc3_units != 0
                else fc2_units
            ),
            action_size,
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if hasattr(self, "fc3"):
            x = F.relu(self.fc3(x))
        if hasattr(self, "fc4"):
            x = F.relu(self.fc4(x))
        return self.fc5(x)
