import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

#torch_model = QNetwork(8, 4, 0)
#torch.onnx.export(torch_model,               # model being run
#        torch.tensor([-0.0113382, 1.4284717,  -0.5734692, 0.37714776, 0.01300366, 0.12856543, 0., 0.]),                 
#                 "DQN.onnx",
#                  export_params=False,
#                  opset_version=10, 
#                  do_constant_folding=False,
#                  input_names = ['input'],   # the model's input names
#                  output_names = ['output'], # the model's output names
#                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                'output' : {0 : 'batch_size'}})

