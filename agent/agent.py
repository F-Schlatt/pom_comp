import os

import numpy as np
import torch

from arm import Policy
from pommerman.agents import BaseAgent

from helpers import featurize_obs, center_boards
from network import Network, VIEW_DISTANCE


class Agent(BaseAgent):
    """Agent based on Advantage Regret Minimization
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super(Agent, self).__init__(*args, **kwargs)

        dir_path = os.path.dirname(os.path.realpath(__file__))

        state_dict = torch.load('{}/agent.pth'.format(dir_path))

        network = Network()
        network.load_state_dict(state_dict)

        self.policy = Policy(network)

    def act(self, obs, action_space):
        img, scalar = featurize_obs(obs)
        inp = img
        player = inp[0]
        player_pos = np.nonzero(player)
        inp = center_boards(inp, VIEW_DISTANCE, player_pos)

        inp = inp[1:]

        inp = inp.reshape(-1)

        inp = np.concatenate((inp, scalar))
        inp = torch.tensor(inp).float().unsqueeze(0)

        action = self.policy(inp)

        return action
