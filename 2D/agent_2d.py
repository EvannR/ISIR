import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class Agent:
    """
    Agent class representing an individual agent in the grid environment.
    Tracks meal history and calculates satisfaction/emotion levels.
    """
    
    def __init__(self, agent_id, memory_size=10, beta=0.5, threshold=0.5, 
                 smoothing='linear', sigmoid_gain=10.0):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_size: Size of meal history buffer
            beta: Weight of last meal vs history for satisfaction (0-1)
            threshold: Consumption rate threshold for neutral emotion
            smoothing: 'linear' or 'sigmoid' for emotion mapping
            sigmoid_gain: Steepness for sigmoid mapping
        """
        self.agent_id = agent_id
        self.memory_size = memory_size
        self.beta = beta
        self.threshold = threshold
        self.smoothing = smoothing
        self.sigmoid_gain = sigmoid_gain
        
        # Meal history: deque of 0s and 1s (0=no meal, 1=meal)
        self.meal_history = deque([0] * memory_size, maxlen=memory_size)
        self.total_meals = 0
        
    def record_meal(self, success: bool, reward: float):
        """
        Record whether the agent successfully consumed a resource.
        
        Args:
            success: True if agent successfully consumed resource
            reward: Reward received (for tracking, not used here)
        """
        self.meal_history.append(1 if success else 0)
        if success:
            self.total_meals += 1
    
    def get_recent_meals(self):
        """Return count of recent meals in history."""
        return sum(self.meal_history)
    
    def _consumption_rate(self):
        """Calculate consumption rate from meal history [0, 1]."""
        if len(self.meal_history) == 0:
            return 0.0
        return sum(self.meal_history) / len(self.meal_history)
    
    def _linear_emotion(self, rate):
        """Linear mapping from consumption rate to emotion [-1, 1]."""
        t = self.threshold
        if rate >= t:
            return (rate - t) / (1 - t) if t < 1 else 1.0
        return -(t - rate) / t if t > 0 else -1.0
    
    def _sigmoid_emotion(self, rate):
        """Sigmoid mapping from consumption rate to emotion [-1, 1]."""
        g = self.sigmoid_gain
        t = self.threshold
        if rate >= t:
            norm = (rate - t) / (1 - t) if t < 1 else 1.0
        else:
            norm = (rate - t) / t if t > 0 else -1.0
        exp_val = np.exp(-g * norm)
        return 2.0 / (1.0 + exp_val) - 1.0
    
    def get_emotion(self):
        """
        Calculate emotion signal from consumption rate [-1, 1].
        -1 = very unsatisfied, 0 = neutral, 1 = very satisfied
        """
        rate = self._consumption_rate()
        if self.smoothing == 'sigmoid':
            return self._sigmoid_emotion(rate)
        return self._linear_emotion(rate)
    
    def get_satisfaction(self):
        """
        Calculate personal satisfaction [0, 1] from last meal and history.
        This is used for empathic reward calculation.
        """
        if len(self.meal_history) == 0:
            return 0.0
        
        last = 1.0 if self.meal_history[-1] > 0 else 0.0
        hist = sum(self.meal_history) / len(self.meal_history)
        return self.beta * last + (1 - self.beta) * hist
    
    def reset(self):
        """Reset agent state for a new episode."""
        self.meal_history = deque([0] * self.memory_size, maxlen=self.memory_size)
        self.total_meals = 0


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class EmpathicRewardCalculator:
    """
    Reward shaping helper.

    This keeps the *formula* close to the agent-side concepts (satisfaction/emotion),
    while the environment stays responsible for providing neighborhood information.

    Parameters:
    - exploit_reward: maximum reward when consuming
    - empathic_penalty: minimum reward (maximum punishment) when consuming
    - empathy_alpha: weight on others' emotion vs. personal satisfaction (0..1)
    """

    def __init__(self, exploit_reward: float, empathic_penalty: float, empathy_alpha: float):
        self.exploit_reward = float(exploit_reward)
        self.empathic_penalty = float(empathic_penalty)
        self.empathy_alpha = float(empathy_alpha)
        if self.empathy_alpha < 0.0:
            self.empathy_alpha = 0.0
        if self.empathy_alpha > 1.0:
            self.empathy_alpha = 1.0

    def consumption_reward(self, *, self_agent, nearby_agents):
        """
        Compute the *consumption reward* term to add when `self_agent` consumes a resource.

        If there are no nearby agents, returns exploit_reward.
        """
        if not nearby_agents:
            return self.exploit_reward

        personal = float(self_agent.get_satisfaction())  # [0,1]
        hungriness = 1 - personal 
        others_emo = float(np.mean([a.get_emotion() for a in nearby_agents]))  # [-1,1]

        alpha = self.empathy_alpha
        raw = (1.0 - alpha) * hungriness + alpha * others_emo  # approx [-alpha, 1]

        # Unified normalization for all alpha: normalize raw to allow negative values
        # raw ranges from approximately -alpha (when hungriness=0, others_emo=-1) to 1 (when hungriness=1, others_emo=1)
        # Use raw / (1.0 + alpha) to preserve general shape while allowing negatives when others_emo < 0
        norm = raw / (1.0 + alpha) if (1.0 + alpha) != 0 else raw
        
        # Don't clip - allow negative values for negative neighbor emotions
        # Use exploit_reward * norm: ensures non-negative when hungriness=0 and others_emo >= 0,
        # while allowing negative rewards when others_emo < 0
        return float(self.exploit_reward * norm)

    # Backward-compatible alias: older code calls this "bonus".
    def consumption_bonus(self, *, self_agent, nearby_agents):
        """
        Alias for `consumption_reward` (kept for backward compatibility).
        """
        return self.consumption_reward(self_agent=self_agent, nearby_agents=nearby_agents)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        if isinstance(action, (list, tuple)):
            action = np.array([action[0]], dtype=np.int64)
        elif isinstance(action, np.ndarray):
            action = np.array([action.item()], dtype=np.int64)
        else:
            action = np.array([int(action)], dtype=np.int64)
        
        state = np.array(state, dtype=np.float32)
        if not isinstance(reward, (int, float)):
            reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32)
        if not isinstance(done, bool):
            done = bool(done)
        
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size, device=None):
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.from_numpy(np.vstack([e.state.flatten() for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.tensor([[e.reward] for e in experiences], dtype=torch.float)
        next_states = torch.from_numpy(np.vstack([e.next_state.flatten() for e in experiences])).float()
        dones = torch.tensor([[int(e.done)] for e in experiences], dtype=torch.float)
        
        # Move tensors to device if specified
        if device is not None:
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for grid navigation and resource exploitation.
    Takes flattened observation as input and outputs Q-values for each action.
    """
    
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_dim)
        
        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    """
    DQN agent with experience replay and target network.
    """
    
    def __init__(self, agent_id, state_size, action_size, 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=32,
                 hidden_size=128, update_target_every=100,
                 memory_size=10, beta=0.5, threshold=0.5,
                 smoothing='linear', sigmoid_gain=10.0):
        """
        Initialize DQN agent.
        
        Args:
            agent_id: Unique identifier
            state_size: Shape of state (will be flattened)
            action_size: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per step
            epsilon_min: Minimum epsilon
            batch_size: Batch size for training
            hidden_size: Hidden layer size
            update_target_every: Frequency of target network updates
            memory_size: Meal history buffer size
            beta: Satisfaction calculation parameter
            threshold: Emotion threshold
            smoothing: Emotion smoothing type
            sigmoid_gain: Sigmoid gain parameter
        """
        self.agent = Agent(agent_id, memory_size, beta, threshold, smoothing, sigmoid_gain)
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0
        
        # Calculate flattened state dimension
        if isinstance(state_size, tuple):
            self.state_dim = int(np.prod(state_size))
        else:
            self.state_dim = state_size
        
        # Device detection (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if agent_id == 0:  # Only print once per agent group
            print(f"Agent {agent_id} using device: {self.device}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # Networks - move to device
        self.policy_network = DQNNetwork(self.state_dim, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=50000)
        
        # Current state tracking
        self.current_state = None
        self.previous_action = None
    
    def select_action(self, state, training=True, valid_actions=None):
        """
        Select action using epsilon-greedy policy with optional action masking.
        
        Args:
            state: Current observation
            training: If False, always use greedy policy
            valid_actions: Optional list of valid action indices. If provided, only
                          these actions can be selected (action masking).
        
        Returns:
            Selected action (int)
        """
        # Filter valid actions if provided
        if valid_actions is not None:
            if len(valid_actions) == 0:
                # Fallback: if no valid actions provided, allow all
                valid_actions = list(range(self.action_size))
            valid_actions = list(valid_actions)  # Ensure it's a list
        else:
            # No masking: all actions are valid
            valid_actions = list(range(self.action_size))
        
        # Epsilon-greedy exploration: choose randomly from valid actions
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))
        
        # Greedy policy: choose best action from valid actions only
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        state_flat = state.flatten()
        state_tensor = torch.from_numpy(state_flat).float().unsqueeze(0).to(self.device)
        
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state_tensor)
        self.policy_network.train()
        
        # Apply action masking: set invalid actions to -inf
        action_values_np = action_values.cpu().data.numpy().flatten()
        masked_values = np.full(self.action_size, -np.inf)
        masked_values[valid_actions] = action_values_np[valid_actions]
        
        # Select best valid action
        return int(np.argmax(masked_values))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(reward[0])
        else:
            reward = float(reward)
        if isinstance(done, (list, tuple, np.ndarray)):
            done = bool(done[0])
        else:
            done = bool(done)
        
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """Train the agent on a batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
        
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
    
    def step(self, next_state, reward, done, valid_actions=None):
        """
        Execute one step: remember experience, learn, update state.
        
        Args:
            next_state: Next observation
            reward: Reward received
            done: Whether episode is done
            valid_actions: Optional list of valid action indices for action masking
        
        Returns:
            Selected action for next step
        """
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        # Remember experience if we have previous state
        if self.current_state is not None and self.previous_action is not None:
            self.remember(self.current_state, self.previous_action, reward, next_state, done)
            self.learn()
        
        # Update state and select next action (with action masking if provided)
        self.current_state = next_state
        action = self.select_action(next_state, training=True, valid_actions=valid_actions)
        self.previous_action = action
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return action
    
    def reset(self, observation=None):
        """Reset agent for new episode."""
        self.agent.reset()
        self.current_state = observation
        self.previous_action = None
        if observation is not None:
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
            self.current_state = observation
    
    def record_meal(self, success: bool, reward: float):
        """Record meal in agent's history."""
        self.agent.record_meal(success, reward)
    
    def get_satisfaction(self):
        """Get agent's current satisfaction level."""
        return self.agent.get_satisfaction()
    
    def get_emotion(self):
        """Get agent's current emotion level."""
        return self.agent.get_emotion()

