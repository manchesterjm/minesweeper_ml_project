"""
Deep Q-Network (DQN) agent for Minesweeper.

Uses PyTorch with CUDA support for GPU acceleration.
Implements Double DQN with prioritized experience replay.
"""
from typing import Optional, Tuple, List
from dataclasses import dataclass
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent


# ============================================================================
# Device Configuration
# ============================================================================

def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Neural Network Architecture
# ============================================================================

class DQNNetwork(nn.Module):
    """
    Convolutional neural network for Q-value estimation.

    Architecture:
        - 3 convolutional layers with batch normalization
        - 2 fully connected layers
        - Dueling architecture (value + advantage streams)
    """

    def __init__(
        self,
        height: int,
        width: int,
        n_actions: int,
    ) -> None:
        """
        Initialize the DQN network.

        Args:
            height: Board height.
            width: Board width.
            n_actions: Number of possible actions.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_actions = n_actions

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate flattened size
        conv_out_size = height * width * 128

        # Dueling architecture: separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, height, width).

        Returns:
            Q-values for each action, shape (batch, n_actions).
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Dueling: Q = V + (A - mean(A))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


# ============================================================================
# Experience Replay Buffer
# ============================================================================

@dataclass
class Experience:
    """Single experience tuple for replay."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer with uniform sampling.

    Stores transitions and provides random mini-batches for training.
    """

    def __init__(self, capacity: int = 100000) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store.
        """
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences."""
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with Double DQN and experience replay.

    Features:
        - CUDA/GPU acceleration when available
        - Double DQN for reduced overestimation
        - Dueling architecture for better value estimation
        - Epsilon-greedy exploration with decay
        - Experience replay for stable learning
    """

    def __init__(
        self,
        board_height: int = 9,
        board_width: int = 9,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the DQN agent.

        Args:
            board_height: Number of rows in the board.
            board_width: Number of columns in the board.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor for future rewards.
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Epsilon decay rate per step.
            buffer_size: Replay buffer capacity.
            batch_size: Mini-batch size for training.
            target_update_freq: Steps between target network updates.
            device: Torch device (auto-detected if None).
        """
        super().__init__(board_height, board_width)

        self.device = device or get_device()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Initialize networks
        n_actions = board_height * board_width
        self.policy_net = DQNNetwork(
            board_height, board_width, n_actions
        ).to(self.device)
        self.target_net = DQNNetwork(
            board_height, board_width, n_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.losses: List[float] = []

    def select_action(
        self,
        observation: np.ndarray,
        valid_actions: Optional[np.ndarray] = None,
        training: bool = True,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            observation: 2D array of cell states.
            valid_actions: Optional mask of valid actions.
            training: Whether to use exploration (epsilon-greedy).

        Returns:
            Selected action index.
        """
        if valid_actions is None:
            valid_actions = self.get_valid_actions_from_obs(observation)

        valid_indices = np.where(valid_actions)[0]

        if len(valid_indices) == 0:
            return 0

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_indices)

        # Greedy action selection
        with torch.no_grad():
            state = self._preprocess_observation(observation)
            q_values = self.policy_net(state).cpu().numpy()[0]

            # Mask invalid actions with very negative values
            masked_q = np.full_like(q_values, -np.inf)
            masked_q[valid_indices] = q_values[valid_indices]

            return int(np.argmax(masked_q))

    def _preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Convert numpy observation to tensor."""
        # Normalize observation to [-1, 1] range
        obs_normalized = observation.astype(np.float32) / 9.0
        tensor = torch.from_numpy(obs_normalized).unsqueeze(0)
        return tensor.to(self.device)

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store experience and train on mini-batch.

        Args:
            observation: State before action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after action.
            done: Whether episode ended.
        """
        # Store experience
        self.replay_buffer.push(
            observation, action, reward, next_observation, done
        )

        self.steps += 1

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end, self.epsilon * self.epsilon_decay
        )

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Train if enough experiences
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

    def _train_step(self) -> None:
        """Perform one training step on a mini-batch."""
        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare batch tensors
        states = torch.stack([
            self._preprocess_observation(e.state).squeeze(0)
            for e in batch
        ])
        actions = torch.tensor(
            [e.action for e in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [e.reward for e in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([
            self._preprocess_observation(e.next_state).squeeze(0)
            for e in batch
        ])
        dones = torch.tensor(
            [e.done for e in batch], dtype=torch.float32, device=self.device
        )

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = F.smooth_l1_loss(current_q.squeeze(1), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())

    def save(self, path: str) -> None:
        """Save model weights to file."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

    def reset(self) -> None:
        """Reset episode-specific state."""
        pass

    @property
    def device_name(self) -> str:
        """Get device name for logging."""
        return str(self.device)
