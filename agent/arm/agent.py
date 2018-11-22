import numpy as np
import torch
from pommerman.agents import BaseAgent

from .helpers import featurize_obs

from .arm import Arm

import os 

class Agent(BaseAgent):
    """Agent based on Advantage Regret Minimization
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super(Agent, self).__init__(*args, **kwargs)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.arm_net = Arm('%s/agent.model' % (dir_path))

        self.save_board = np.zeros((11, 11))

    def episode_end(self, reward):
        self.save_board = np.zeros((11, 11))

    def act(self, obs, action_space):

        images, scalar, save_board = featurize_obs(obs, self.save_board)

        self.save_board = save_board

        actions = np.eye(action_space.n)

        images = torch.tensor(images).unsqueeze(0).float()
        scalar = torch.tensor(scalar).unsqueeze(0).float()
        actions = [torch.tensor(action).unsqueeze(0).float()
                   for action in actions]

        action = self.arm_net.choose_action(images, scalar, actions)

        return action