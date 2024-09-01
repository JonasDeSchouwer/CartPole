import gym
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.transforms import ToTensor


class cartNet(torch.nn.Module):
    def __init__(self):
        super(cartNet, self).__init__()

        self.fcl1 = nn.Linear(5, 50)
        self.fcl2 = nn.Linear(50,50)
        self.fcl3 = nn.Linear(50,1)

    def forward(self, state):
        """
        return the Q-value for a given state x
        :param state: Bx5 matrix:
            first 4 elements: observation
            last element: action
        """
        x = torch.FloatTensor(state)
        x = self.fcl1(x)
        x = F.relu(x)
        x = self.fcl2(x)
        x = F.relu(x)
        x = self.fcl3(x)
        x = F.relu(x)

        return x
    
    def select_action(self, observation, possible_actions, lmbda=np.inf):
        """
        :param lmbda:
            a parameter that controls the explore/exploit behavior of the agent,
            lmbda = inf is 100% exploit
            lmbda = 0 is 100% random
            lmbda = -inf selects the action with lowest q-value
        """
        state = torch.zeros(len(observation)+1)
        state[:len(observation)] = torch.FloatTensor(list(observation))

        actions_q = torch.zeros(len(possible_actions))
        for i, action in enumerate(possible_actions):
            state[-1] = action
            actions_q[i] = self.forward(state).detach()
        
        if lmbda == np.inf:
            return int(np.argmax(actions_q))
        elif lmbda == -np.inf:
            return int(np.argmin(actions_q))
        else:
            actions_exp = torch.exp(actions_q * lmbda)
            actions_probs = actions_exp / sum(actions_exp)
            return int(np.random.choice(possible_actions, p=actions_probs))


    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    
