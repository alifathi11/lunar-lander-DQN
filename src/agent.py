import random
from collections import deque, namedtuple

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np 

from src.model import create_policy_net, create_target_net

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity=100_000): 
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)

        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=100_000,
        batch_size=64,
        device=None
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size

        if device is None: 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = torch.device(device)
        
        self.policy_net = create_policy_net(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.target_net = create_target_net(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.replayBuffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, eps):
        if random.random() < eps: 
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def update(self):
        if len(self.replayBuffer) < self.batch_size:
            return None 

        states, actions, rewards, next_states, dones = self.replayBuffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.int64, device=self.device).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        return loss.item()

    def store_transition(self, state, action, reward, next_state, done): 
        self.replayBuffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path): 
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path): 
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.update_target_network()
    
        