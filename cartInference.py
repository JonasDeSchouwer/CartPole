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

from cartNet import cartNet

# setting environment
env = gym.make('CartPole-v0')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
possible_actions = list(range(env.action_space.n))

plt.ion()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on gpu")
else:
    print("running on cpu")
    device = torch.device("cpu")
net = cartNet().to(device)
net.load("networks/cartNet")
net.eval()

want_to_exit = False
want_to_restart = False
paused = False
def key_press(key,mod):
    global paused, want_to_exit, want_to_restart
    print(key)
    if key == 32:
        paused = not paused
        if paused: print("space key pressed, game is paused")
        else: print("space key pressed, game is unpaused")
    if key == 65307:
        want_to_exit = True
        print("escape key pressed, exiting")
    if key == 114:
        want_to_restart = True
        print("R pressed, restarting")


def main():
    global paused, want_to_exit, want_to_restart
    
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press

    while True:
        want_to_restart = False
        observation = env.reset()
        env.render()
        for i in range(7):
            if want_to_exit: break
            if want_to_restart: break
            time.sleep(0.1)

        if want_to_exit: break
        if want_to_restart: continue

        done = False
        while not done:
            env.render()
            if not paused:
                action = net.select_action(observation, possible_actions, lmbda=np.inf)
                observation, reward, done, _ = env.step(action)
                time.sleep(0.01)
            if want_to_exit:
                print("break")
                break
            if want_to_restart:
                print("restart")
                break
            
    env.close()

main()
