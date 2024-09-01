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

# setting environment
env = gym.make('CartPole-v0')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(env.action_space)

observations = []
env.reset()
for _ in range(1000):
    env.render()
    observation, reward, done, _ = env.step(env.action_space.sample())
    print(observation)
    observations.append(observation)
    time.sleep(1)
    if done:
        break
env.close()