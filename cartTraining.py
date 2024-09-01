from typing import Deque
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


# device
if torch.cuda.is_available():
    print("running on GPU")
    device = torch.device("cuda")
else:
    print("running on CPU")
    device = torch.device("cpu")

# network
cartnet = cartNet().to(device)
cartnet = cartnet.float()

# hyper parameters
EPOCHS = 300
LEARNING_RATE = 0.005
BATCH_SIZE = 20
MEMORY = 3000
MEMORY_RENEWAL = int(1/4 * MEMORY)
DISCOUNT = 0.9
criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(cartnet.parameters(), lr=LEARNING_RATE, momentum=0.9)


# setting environment
env = gym.make('CartPole-v0')
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
possible_actions = list(range(env.action_space.n))


class Transition:
    def __init__(self, state, action, expected_reward=0):
        self.state = np.zeros(len(state)+1)
        self.state[:len(state)] = state
        self.state[-1] = action
        self.expected_reward = expected_reward
   
    def add_expected_reward(self, reward):
        self.expected_reward = torch.FloatTensor([reward])

class Memory:
    # saves state-action-reward (star) as a Nx6 matrix: first 4 columns (obs), next column (act), last column (rew)

    def __init__(self, capacity):
        self.star = torch.zeros((capacity, 6))
        self.length = 0
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.star[index]
    
    def get_stars(self):
        return self.star[:self.length]

    def get_states(self):
        return self.star[:self.length, :4]
    
    def get_actions(self):
        return self.star[:self.length, 4]
    
    def get_rewards(self):
        return self.star[:self.length, 5]

    def add_state(self, state, action, expected_reward=0):
        self.star[self.length][:4] = torch.FloatTensor(state)
        self.star[self.length][4] = action
        self.star[self.length][5] = expected_reward
        self.length += 1

    def add_star(self, star):
        self.star[self.length] = torch.FloatTensor(star)
        self.length += 1
    
    def remove_first_states(self, n):
        np.roll(self.star, -n, axis=0)
        self.star[-n:] = 0
        self.length -= n

    def sample(self, batch_size):
        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:,:5], batch[:,5])     # yield states and rewards per batch as seperate matrices

    


memory = Memory(2*MEMORY)


def run_and_save_episode(net):
    global memory

    # run the environment with the current neural network and save the results to memory
    # return the duration of the episode
    observation = env.reset()
    stars = [] #list of transitions
    rewards = []
    done = False
    while not done:
        action = net.select_action(observation, possible_actions, epoch/(EPOCHS-1))
        stars.append(list(observation) + [action])
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
    
    assert len(stars) == len(rewards)
    n = len(rewards)
    cumulative_reward = 0
    for i in range(n-1,-1,-1):
        cumulative_reward = rewards[i] + DISCOUNT * cumulative_reward
        stars[i] += [cumulative_reward]
    
    for star in stars:
        memory.add_star(star)

    return n


def train_model(net):
    # do one training cycle
    # return the total loss
    running_loss = 0
    for states, rewards in memory.sample(BATCH_SIZE):
        q_values = net(states)
        loss = criterion(q_values, rewards.view(-1,1))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss


data = []   # tuples of (epoch, duration)
for epoch in range(EPOCHS):
    print(f"starting epoch {epoch+1}/{EPOCHS}")

    if len(memory) >= MEMORY:
        memory.remove_first_states(MEMORY_RENEWAL)

    while len(memory) < MEMORY:
        duration = run_and_save_episode(cartnet)
        data.append((epoch, duration))


    loss = train_model(cartnet)
    print(loss)

epochs, durations = zip(*data)
epochs = np.array(epochs)
durations = np.array(durations)

plt.scatter(epochs, durations, c='r')

epochs_avg_duration = list(range(EPOCHS))
avg_duration = np.zeros(EPOCHS)       # avg_duration[i] = gemiddelde tijd in epoch i
for epoch in range(EPOCHS):
    elements = durations[epochs==epoch]
    if len(elements) == 0:
        avg_duration[epoch] = avg_duration[epoch-1]
    else:
        average_dur = np.mean(elements)     #get those columns for which the first element is epoch
        avg_duration[epoch] = average_dur

# take a rolling mean of avg_duration
D = 10
smooth_duration = np.zeros(EPOCHS)
for i in range(EPOCHS):
    start = max(i-D, 0)
    end = min(i+D,EPOCHS)
    smooth_duration[i] = np.mean(avg_duration[start:end])

plt.plot(smooth_duration, 'b')


plt.show()

cartnet.save("networks/cartNet")
    
