import torch

IMG_DIM = 14
SCALAR_DIM = 11
VIEW_DISTANCE = 4
ACTION_DIM = 6

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.relu = torch.nn.ReLU()

        self.board_size = VIEW_DISTANCE * 2 + 1

        self.linear1 = torch.nn.Linear(
            self.board_size * self.board_size * (IMG_DIM-1) + SCALAR_DIM, 2000)
        self.linear2 = torch.nn.Linear(2000, 2000)
        self.linear3 = torch.nn.Linear(2000, 2000)
        self.linear4 = torch.nn.Linear(2000, ACTION_DIM + 1)

        self.device = torch.device('cpu')

    def forward(self, obs):
        out = obs
        out = self.relu(self.linear1(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.linear4(out)
        return out
