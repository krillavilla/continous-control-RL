import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import ReplayBuffer, OUNoise  # Use relative import for utils module

# Check for CUDA availability and handle compatibility issues
try:
    # Attempt to initialize CUDA and check if it's available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, using CPU instead")
except Exception as e:
    # If there's any error with CUDA initialization, fall back to CPU
    device = torch.device("cpu")
    print(f"Error initializing CUDA: {e}")
    print("Falling back to CPU")

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with small values."""
        self.fc1.weight.data.uniform_(*self._hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self._hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def _hidden_init(self, layer):
        """Initialize hidden layer weights."""
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with small values."""
        self.fc1.weight.data.uniform_(*self._hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self._hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def _hidden_init(self, layer):
        """Initialize hidden layer weights."""
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        xs = F.relu(self.bn1(self.fc1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed,
                 buffer_size=int(1e6), batch_size=64, gamma=0.99, tau=5e-3,
                 lr_actor=3e-4, lr_critic=3e-3, weight_decay=0,
                 fc1_units=256, fc2_units=128):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay for critic
            fc1_units (int): number of nodes in first hidden layer
            fc2_units (int): number of nodes in second hidden layer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # Initialize Replay Memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
        
        # Initialize Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        # Initialize noise decay parameters
        self.noise_scale = 1.0
        self.noise_decay = 0.995

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            # Use 2 learning updates per step for faster convergence in early training
            n_updates = 2 if len(self.memory) < 10000 else 1
            for _ in range(n_updates):
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise_scale * self.noise.sample()
            # Decay noise scale for each action
            self.noise_scale *= self.noise_decay

        return np.clip(action, -1, 1)

    def reset(self):
        """Reset the noise process."""
        self.noise.reset()

    def learn(self, experiences, gamma, n_updates=1):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
    
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            n_updates (int): number of times to update networks (for faster learning)
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Apply gradient clipping to critic network
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------ார்கள் update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filepath='checkpoint.pth'):
        """Save the model parameters."""
        torch.save({
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath='checkpoint.pth'):
        """Load the model parameters."""
        checkpoint = torch.load(filepath)
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Copy parameters to target networks
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)