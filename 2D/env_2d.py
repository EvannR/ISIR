import numpy as np
import gymnasium as gym
from collections import deque
from agent_2d import EmpathicRewardCalculator


class EmpathicGridEnv(gym.Env):
    """
    Multi-agent grid environment where agents learn to navigate and share resources empathetically.
    Each agent sees a square around itself on the grid and receives negative reward when taking
    a resource if it's more satisfied than nearby agents.
    """
    
    def __init__(self,
                 grid_size=10,
                 num_agents=6,
                 num_resources=15,
                 vision_radius=2,
                 initial_satisfaction=0.0,
                 beta=0.5,
                 threshold=0.5,
                 smoothing='linear',
                 sigmoid_gain=10.0,
                 memory_size=10,
                 exploit_reward=1.0,
                 empathic_penalty=-2.0,
                 empathy_alpha=0.5,
                 step_cost=-0.01,
                 loop_penalty=0.0,
                 loop_max_pattern_length=6,
                 loop_min_repetitions=4,
                 max_steps=1000,
                 regeneration_mode='always',
                 render_mode=None):
        """
        Initialize the empathic grid environment.

        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            num_agents: Number of agents in the environment
            num_resources: Number of resources on the grid
            vision_radius: Radius of vision (agent sees (2*vision_radius+1) x (2*vision_radius+1) square)
            initial_satisfaction: Initial satisfaction level for agents
            beta: Weight of last meal vs history for satisfaction calculation
            threshold: Consumption rate threshold for neutral emotion
            smoothing: 'linear' or 'sigmoid' for emotion mapping
            sigmoid_gain: Steepness for sigmoid mapping
            memory_size: Size of meal history buffer
            exploit_reward: Reward for successfully exploiting a resource
            empathic_penalty: Penalty when exploiting while more satisfied than nearby agents
            step_cost: Cost per step (encourages efficiency)
            regeneration_mode: Resource respawn mode:
                - 'always': Resources always respawn after consumption (probability = 1)
                - 'dynamic': Respawn probability depends on current resource count
                             (fewer resources = lower respawn probability)
                             Formula: prob = current_count / initial_count
            render_mode: Optional rendering mode
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.vision_radius = vision_radius
        self.vision_size = 2 * vision_radius + 1
        self.memory_size = memory_size
        self.beta = beta
        self.threshold = threshold
        self.smoothing = smoothing
        self.sigmoid_gain = sigmoid_gain
        self.exploit_reward = exploit_reward
        self.empathic_penalty = empathic_penalty
        # Empathy shaping weight (matches game-theoretic SocialRewardCalculator idea):
        # alpha = weight on others' emotional state vs. personal satisfaction (0-1).
        self.empathy_alpha = float(empathy_alpha)
        self.step_cost = step_cost
        # Loop penalty (reward shaping)
        # Loop definition (strict, matches notebooks):
        # - a pattern of at least 2 actions
        # - repeated more than 3 times (>=4 occurrences)
        self.loop_penalty = float(loop_penalty)
        self.loop_max_pattern_length = int(loop_max_pattern_length)
        self.loop_min_repetitions = int(loop_min_repetitions)
        self.max_steps = max_steps
        self.regeneration_mode = regeneration_mode
        self.render_mode = render_mode

        self._reward_calc = EmpathicRewardCalculator(
            exploit_reward=self.exploit_reward,
            empathic_penalty=self.empathic_penalty,
            empathy_alpha=self.empathy_alpha,
        )

        # Clamp alpha for safety
        if self.empathy_alpha < 0.0:
            self.empathy_alpha = 0.0
        if self.empathy_alpha > 1.0:
            self.empathy_alpha = 1.0
        
        # Observation space: vision_size x vision_size grid with channels:
        # Channel 0: Terrain (-1=out of bounds, 0=empty, 1=resource, 2=agent)
        # Channel 1: Satisfaction levels of agents in view (0-1, normalized)
        # Channel 2: Agent's own satisfaction (broadcasted, for self-awareness)
        self.observation_space = gym.spaces.Box(
            low=-1.0, 
            high=2.0, 
            shape=(self.vision_size, self.vision_size, 3), 
            dtype=np.float32
        )
        
        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)
        
        # Agent positions, resources, and agent objects
        self.agent_positions = []
        self.resource_positions = []
        self.agents = []  # Will be set externally to Agent objects
        
        # Random number generator
        self.np_random = np.random.default_rng()
        self.steps_taken = 0

        # Per-agent action history for loop detection
        history_len = max(0, self.loop_max_pattern_length) * max(0, self.loop_min_repetitions) * 2
        self._action_histories = [deque(maxlen=max(8, history_len)) for _ in range(self.num_agents)]

    def _detect_action_loop(self, actions):
        """
        Detect whether the tail of `actions` ends with a repeating pattern.

        Strict loop definition:
        - pattern length >= 2
        - repeated >= self.loop_min_repetitions times consecutively at the end
        """
        if self.loop_min_repetitions <= 3:
            # Enforce "more than 3 times" requirement even if misconfigured.
            min_reps = 4
        else:
            min_reps = self.loop_min_repetitions

        if self.loop_max_pattern_length < 2:
            return False

        if len(actions) < 2 * min_reps:
            # Need at least 2 actions per pattern * min repetitions
            return False

        max_len = min(self.loop_max_pattern_length, len(actions) // min_reps)
        for pattern_len in range(2, max_len + 1):
            if len(actions) < pattern_len * min_reps:
                continue

            last_pattern = actions[-pattern_len:]
            # Count consecutive repeats of last_pattern at the end
            max_possible = len(actions) // pattern_len
            reps = 1
            for r in range(2, max_possible + 1):
                start_idx = len(actions) - r * pattern_len
                if actions[start_idx:start_idx + pattern_len] == last_pattern:
                    reps = r
                else:
                    break

            if reps >= min_reps:
                return True

        return False
        
    def reset(self, seed=None, options=None):
        """Reset the environment at the start of an episode."""
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # Randomly place agents on distinct positions
        self.agent_positions = []
        while len(self.agent_positions) < self.num_agents:
            pos = tuple(self.np_random.integers(0, self.grid_size, size=2))
            if pos not in self.agent_positions:
                self.agent_positions.append(pos)
        
        # Randomly place resources on distinct positions (not colliding with agents)
        self.resource_positions = []
        while len(self.resource_positions) < self.num_resources:
            pos = tuple(self.np_random.integers(0, self.grid_size, size=2))
            if pos not in self.agent_positions and pos not in self.resource_positions:
                self.resource_positions.append(pos)
        
        # Reset all agents (meal history, satisfaction, etc.)
        for agent in self.agents:
            agent.reset()

        # Reset loop detection histories
        self._action_histories = [deque(maxlen=h.maxlen) for h in self._action_histories]
        
        # Return initial observations for all agents
        observations = [self._get_obs(agent_idx) for agent_idx in range(self.num_agents)]
        return observations, {}
    
    def step(self, actions):
        """
        Execute one time-step for all agents.
        
        Args:
            actions: List of actions for each agent (0-3: up, down, left, right)
                     Resources are automatically consumed when agent moves onto them
        
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            done: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional info dictionary
        """
        self.steps_taken += 1
        rewards = [0.0] * self.num_agents
        consumed_flags = [False] * self.num_agents
        consumption_bonus = [0.0] * self.num_agents
        empathic_penalty_events = [False] * self.num_agents

        # Update action histories and pre-compute which agents are looping
        loop_flags = [False] * self.num_agents
        if self.loop_penalty != 0.0:
            for i, action in enumerate(actions):
                self._action_histories[i].append(int(action))
                loop_flags[i] = self._detect_action_loop(list(self._action_histories[i]))
        
        # Process movements and update positions
        new_positions = list(self.agent_positions)
        occupied_targets = set()
        
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            nx, ny = x, y  # default (stay in place)
            
            if action == 0 and y > 0:  # up
                nx, ny = x, y - 1
            elif action == 1 and y < self.grid_size - 1:  # down
                nx, ny = x, y + 1
            elif action == 2 and x > 0:  # left
                nx, ny = x - 1, y
            elif action == 3 and x < self.grid_size - 1:  # right
                nx, ny = x + 1, y
            
            # Prevent collisions with other agents
            if (nx, ny) in self.agent_positions and (nx, ny) != (x, y):
                nx, ny = x, y
            if (nx, ny) in occupied_targets:
                nx, ny = x, y
            occupied_targets.add((nx, ny))
            
            new_positions[i] = (nx, ny)
        
        # Update agent positions
        self.agent_positions = new_positions
        
        # Process resource consumption and calculate rewards
        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            reward = self.step_cost  # Base step cost
            
            # Check if agent moved onto a resource (automatic consumption)
            if (x, y) in self.resource_positions:
                consumed_flags[i] = True
                # Agent is on a resource - automatically consume it
                
                # Check satisfaction of nearby agents (within vision radius)
                nearby_agent_indices = []
                for j, (ax, ay) in enumerate(self.agent_positions):
                    if j != i:
                        distance = abs(x - ax) + abs(y - ay)  # Manhattan distance
                        if distance <= self.vision_radius:
                            nearby_agent_indices.append(j)

                # Social reward shaping :
                if nearby_agent_indices:
                    bonus = self._reward_calc.consumption_bonus(
                        self_agent=self.agents[i],
                        nearby_agents=[self.agents[j] for j in nearby_agent_indices],
                    )
                    reward += bonus
                    consumption_bonus[i] = bonus
                    # Mark an "empathic penalty event" when consuming was net-negative (bonus < 0).
                    empathic_penalty_events[i] = bonus < 0.0
                else:
                    bonus = self._reward_calc.consumption_bonus(
                        self_agent=self.agents[i],
                        nearby_agents=[],
                    )
                    reward += bonus
                    consumption_bonus[i] = float(bonus)
                    empathic_penalty_events[i] = False

                success = True  # Resource is always consumed when stepped on

                # Remove the consumed resource
                self.resource_positions.remove((x, y))

                # Determine respawn probability based on regeneration mode
                if self.regeneration_mode == 'dynamic':
                    # Fewer resources = lower respawn probability
                    # prob = current_count / initial_count
                    current_count = len(self.resource_positions)
                    respawn_prob = current_count / self.num_resources
                else:
                    # 'always' mode: probability = 1
                    respawn_prob = 1.0

                # Spawn new resource based on probability
                if self.np_random.random() < respawn_prob:
                    empty_spots = [
                        (a, b) for a in range(self.grid_size) for b in range(self.grid_size)
                        if (a, b) not in self.resource_positions
                        and (a, b) not in self.agent_positions
                    ]
                    if empty_spots:
                        new_res = empty_spots[self.np_random.integers(0, len(empty_spots))]
                        self.resource_positions.append(new_res)
                
                # Record meal in agent's history (always True since resource was consumed)
                self.agents[i].record_meal(success, 1.0)
            else:
                # No resource at this position - no meal
                self.agents[i].record_meal(False, 0.0)

            # Loop penalty: discourage repeating action patterns
            if loop_flags[i]:
                reward += self.loop_penalty
            
            rewards[i] = round(reward, 3)
        
        # Check termination conditions
        done = (self.steps_taken >= self.max_steps) or (len(self.resource_positions) == 0)
        
        # Get updated observations
        observations = [self._get_obs(agent_idx) for agent_idx in range(self.num_agents)]
        
        info = {
            'steps': self.steps_taken,
            'resources_remaining': len(self.resource_positions),
            'satisfactions': [agent.get_satisfaction() for agent in self.agents],
            'loop_flags': loop_flags,
            'consumed': consumed_flags,
            'consumption_bonus': consumption_bonus,
            'empathic_penalty_events': empathic_penalty_events
        }
        
        return observations, rewards, done, False, info
    
    def get_valid_actions(self, agent_idx):
        """
        Get list of valid actions for an agent (actions that won't move into walls).
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            List of valid action indices (0-3: up, down, left, right)
        """
        x, y = self.agent_positions[agent_idx]
        valid_actions = []
        
        # Check each direction
        if y > 0:  # Can move up
            valid_actions.append(0)
        if y < self.grid_size - 1:  # Can move down
            valid_actions.append(1)
        if x > 0:  # Can move left
            valid_actions.append(2)
        if x < self.grid_size - 1:  # Can move right
            valid_actions.append(3)
        
        # Always at least one valid action (staying in place if needed, though
        # we don't have a "stay" action, so if all directions blocked, return empty
        # This shouldn't happen in practice, but handle edge case
        if len(valid_actions) == 0:
            # If somehow all directions blocked, allow all (shouldn't happen)
            valid_actions = [0, 1, 2, 3]
        
        return valid_actions
    
    def _get_obs(self, agent_idx):
        """
        Construct observation for agent: a square view around the agent.
        
        Returns:
            (vision_size, vision_size, 3) array:
            - Channel 0: Terrain (-1=out of bounds, 0=empty, 1=resource, 2=agent)
            - Channel 1: Satisfaction levels of other agents in view (0-1)
            - Channel 2: Agent's own satisfaction (broadcasted)
        """
        gx, gy = self.agent_positions[agent_idx]
        obs_terrain = np.zeros((self.vision_size, self.vision_size), dtype=np.float32)
        obs_satisfaction = np.zeros((self.vision_size, self.vision_size), dtype=np.float32)
        obs_own_satisfaction = np.zeros((self.vision_size, self.vision_size), dtype=np.float32)
        
        agent_satisfaction = self.agents[agent_idx].get_satisfaction()
        obs_own_satisfaction.fill(agent_satisfaction)  # Broadcast own satisfaction
        
        # Loop over vision window
        for dx in range(-self.vision_radius, self.vision_radius + 1):
            for dy in range(-self.vision_radius, self.vision_radius + 1):
                obs_x = dx + self.vision_radius
                obs_y = dy + self.vision_radius
                x, y = gx + dx, gy + dy
                
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # Within bounds
                    if (x, y) in self.resource_positions:
                        obs_terrain[obs_y, obs_x] = 1.0  # Resource
                    elif (x, y) in self.agent_positions:
                        obs_terrain[obs_y, obs_x] = 2.0  # Agent
                        # Get satisfaction of agent at this position
                        agent_at_pos_idx = self.agent_positions.index((x, y))
                        if agent_at_pos_idx != agent_idx:  # Other agent
                            obs_satisfaction[obs_y, obs_x] = self.agents[agent_at_pos_idx].get_satisfaction()
                        else:  # Self
                            obs_satisfaction[obs_y, obs_x] = agent_satisfaction
                    else:
                        obs_terrain[obs_y, obs_x] = 0.0  # Empty
                else:
                    # Out of bounds
                    obs_terrain[obs_y, obs_x] = -1.0
                    obs_satisfaction[obs_y, obs_x] = 0.0
        
        # Stack channels
        return np.dstack((obs_terrain, obs_satisfaction, obs_own_satisfaction)).astype(np.float32)
    
    def close(self):
        """Close the environment."""
        pass

