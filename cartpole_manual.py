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


human_action = 0
paused = False
want_to_exit = False
def key_press(key,mod):
    global human_action, paused, want_to_exit
    print(f"key {key} pressed")
    if key == 65363:
        human_action = 1  #right
        print("right key pressed")
    if key == 65361:
        human_action = 0  #left
        print("left key pressed")
    if key == 32:
        paused = not paused
        if paused: print("space key pressed, game is paused")
        else: print("space key pressed, game is unpaused")
    if key == 65307:
        want_to_exit = True
        print("escape key pressed, exiting")

env.render()
env.unwrapped.viewer.window.on_key_press = key_press

while True:
    env.reset()
    env.render()
    for i in range(7):
        if (want_to_exit):
            break
        time.sleep(0.1)

    if want_to_exit: break

    done = False
    while not done:
        env.render()
        if not paused:
            observation, reward, done, _ = env.step(human_action)
            time.sleep(0.1)
        if want_to_exit:
            break
env.close()