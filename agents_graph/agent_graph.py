"""
Graph DQN Agent Module

Adapts the DQN agent for graph-based environment with:
- Flat observation space (no spatial grid)
- Binary action space (wait/consume)
- Imports core components from 2D module
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, List

# Add 2D module to path for imports
_2D_PATH = Path(__file__).parent.parent / '2D'
if str(_2D_PATH) not in sys.path:
    sys.path.insert(0, str(_2D_PATH))

# Import from 2D module (no modifications to original code)
from agent_2d import Agent, EmpathicRewardCalculator, ReplayBuffer, Experience

# Re-export for convenience
__all__ = ['Agent', 'EmpathicRewardCalculator', 'ReplayBuffer', 'Experience', 'GraphDQNAgent']


class GraphDQNNetwork(nn.Module):
    """
    Neural network for graph-based DQN.

    Takes flat observation vector and outputs Q-values for binary actions.
    Simpler architecture than grid-based version due to smaller input.
    """

    def __init__(self, state_dim: int, action_dim: int = 2, hidden_size: int = 64):
        """
        Initialize network.

        Args:
            state_dim: Dimension of flattened state
            action_dim: Number of actions (default 2: wait, consume)
            hidden_size: Size of hidden layers
        """
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_dim)

        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class GraphDQNAgent:
    """
    DQN agent adapted for graph-based environment.

    Key differences from grid-based DQNAgent:
    - Flat observation (no 2D spatial structure)
    - Binary action space (wait/consume)
    - No action masking (both actions always valid)

    Wraps the Agent class for meal history and satisfaction tracking.
    """

    def __init__(
        self,
        agent_id: int,
        observation_dim: int,
        action_size: int = 2,
        learning_rate: float = 0.002,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.05,
        batch_size: int = 32,
        hidden_size: int = 64,
        update_target_every: int = 20,
        replay_capacity: int = 50000,
        # Agent parameters (for meal history tracking)
        memory_size: int = 10,
        beta: float = 0.5,
        threshold: float = 0.5,
        smoothing: str = 'linear',
        sigmoid_gain: float = 10.0
    ):
        """
        Initialize GraphDQNAgent.

        Args:
            agent_id: Unique identifier
            observation_dim: Dimension of flat observation vector
            action_size: Number of actions (default 2)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per step
            epsilon_min: Minimum epsilon
            batch_size: Batch size for training
            hidden_size: Hidden layer size
            update_target_every: Frequency of target network updates
            replay_capacity: Size of replay buffer
            memory_size: Meal history buffer size (for Agent)
            beta: Satisfaction calculation parameter (for Agent)
            threshold: Emotion threshold (for Agent)
            smoothing: Emotion smoothing type (for Agent)
            sigmoid_gain: Sigmoid gain parameter (for Agent)
        """
        # Internal Agent for meal history and satisfaction tracking
        self.agent = Agent(
            agent_id=agent_id,
            memory_size=memory_size,
            beta=beta,
            threshold=threshold,
            smoothing=smoothing,
            sigmoid_gain=sigmoid_gain
        )

        self.agent_id = agent_id
        self.observation_dim = observation_dim
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0

        # Device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if agent_id == 0:
            print(f"GraphDQNAgent using device: {self.device}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Networks
        self.policy_network = GraphDQNNetwork(
            observation_dim, action_size, hidden_size
        ).to(self.device)

        self.target_network = GraphDQNNetwork(
            observation_dim, action_size, hidden_size
        ).to(self.device)

        self.target_network.load_state_dict(self.policy_network.state_dict())

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )
        self.memory = ReplayBuffer(capacity=replay_capacity)

        # State tracking
        self.current_state: Optional[np.ndarray] = None
        self.previous_action: Optional[int] = None

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation (flat vector)
            training: If False, always use greedy policy

        Returns:
            Selected action (0=wait, 1=consume)
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_size))

        # Greedy policy
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        self.policy_network.train()

        return int(action_values.argmax().item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        self.memory.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Train the agent on a batch from replay buffer.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Current Q values
        q_expected = self.policy_network(states).gather(1, actions)

        # Next Q values from target network
        with torch.no_grad():
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return loss.item()

    def step(self, next_state: np.ndarray, reward: float, done: bool) -> int:
        """
        Execute one step: remember experience, learn, update state.

        Args:
            next_state: Next observation
            reward: Reward received
            done: Whether episode is done

        Returns:
            Selected action for next step
        """
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        # Remember experience if we have previous state
        if self.current_state is not None and self.previous_action is not None:
            self.remember(
                self.current_state,
                self.previous_action,
                reward,
                next_state,
                done
            )
            self.learn()

        # Update state and select next action
        self.current_state = next_state
        action = self.select_action(next_state, training=True)
        self.previous_action = action

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

    def reset(self, observation: Optional[np.ndarray] = None) -> None:
        """Reset agent for new episode."""
        self.agent.reset()
        self.current_state = None
        self.previous_action = None

        if observation is not None:
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
            self.current_state = observation

    # ----------------------------------------
    # Delegate to internal Agent
    # ----------------------------------------

    def record_meal(self, success: bool, reward: float) -> None:
        """Record meal in agent's history."""
        self.agent.record_meal(success, reward)

    def get_satisfaction(self) -> float:
        """Get agent's current satisfaction level [0, 1]."""
        return self.agent.get_satisfaction()

    def get_emotion(self) -> float:
        """Get agent's current emotion level [-1, 1]."""
        return self.agent.get_emotion()

    def get_recent_meals(self) -> int:
        """Get count of recent meals in history."""
        return self.agent.get_recent_meals()

    def get_total_meals(self) -> int:
        """Get total meals consumed."""
        return self.agent.total_meals
