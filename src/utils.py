import numpy as np
import random
import copy
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


import numpy as np
import random
import copy
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process with improved exploration."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, min_sigma=0.05, decay_period=10000):
        """Initialize parameters and noise process.
        
        Parameters
        ----------
        size : int
            Size of the action space
        seed : int
            Random seed
        mu : float
            Mean of the noise
        theta : float
            Parameter controlling the speed of mean reversion
        sigma : float
            Initial standard deviation of the noise
        min_sigma : float
            Minimum standard deviation (sigma will decay to this)
        decay_period : int
            Number of steps over which to decay sigma
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.size = size
        self.seed = random.seed(seed)
        self.step = 0
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu) with some randomness."""
        self.state = copy.copy(self.mu)
        # Add some initial randomness to encourage different exploration paths
        self.state += np.random.normal(0, self.sigma, self.size)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # Calculate decayed sigma
        sigma = self.sigma - (self.sigma - self.min_sigma) * min(1.0, self.step / self.decay_period)
        
        # Update state using OU process
        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        
        # Occasionally add a burst of exploration
        if random.random() < 0.01:  # 1% chance
            self.state += np.random.normal(0, 3*sigma, self.size)
            
        self.step += 1
        return self.state


def plot_scores(scores, window_size=100):
    """Plot scores and average scores over time."""
    # Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)

    # Plot rolling average
    rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(len(rolling_mean)) + window_size - 1, rolling_mean)

    # Add horizontal line at 30.0
    plt.axhline(y=30.0, color='r', linestyle='-')

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(['Score', f'Average Score (window={window_size})', 'Target Score (30.0)'])

    return fig
