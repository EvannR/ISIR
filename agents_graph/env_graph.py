"""
Graph-Based Empathic Environment

Multi-agent environment where:
- Agents are fixed nodes in a social graph (no movement)
- Graph edges determine who can see whose emotions
- Agents share a centralized resource pool
- Actions: wait (0) or consume (1)
"""

import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Optional, Any

try:
    # When imported as part of package
    from .social_graph import SocialGraph
    from .resource_pool import ResourcePool
    from .agent_graph import EmpathicRewardCalculator
except ImportError:
    # When run directly or imported without package context
    from social_graph import SocialGraph
    from resource_pool import ResourcePool
    from agent_graph import EmpathicRewardCalculator


class EmpathicGraphEnv(gym.Env):
    """
    Multi-agent graph-based environment for empathic resource sharing.

    Key features:
    - Agents are fixed nodes in a social graph
    - Graph edges define emotional visibility (who sees whose emotions)
    - Shared resource pool instead of spatial resources
    - Binary actions: wait (0) or consume (1)
    - Empathic reward based on neighbors' emotions when consuming
    """

    def __init__(
        self,
        num_agents: int = 5,
        max_neighbors: int = 8,

        # Resource pool settings
        initial_pool: int = 50,
        max_pool_capacity: int = 100,
        regeneration_rate: float = 1.0,
        regeneration_mode: str = 'constant',

        # Graph settings
        graph_type: str = 'random',
        edge_probability: float = 0.4,
        small_world_k: int = 2,
        small_world_p: float = 0.3,
        custom_adjacency: Optional[np.ndarray] = None,
        # Dynamic graph (only when graph_type == 'dynamic')
        dynamic_p_base: float = 0.05,
        dynamic_p_max: float = 0.95,
        dynamic_delta: float = 0.02,
        dynamic_decay: float = 0.9,

        # Agent settings
        memory_size: int = 10,
        beta: float = 0.5,
        threshold: float = 0.5,
        smoothing: str = 'linear',
        sigmoid_gain: float = 10.0,

        # Reward settings
        exploit_reward: float = 1.0,
        empathic_penalty: float = -1.0,
        empathy_alpha: float = 0.5,
        step_cost: float = -0.01,
        wait_reward: float = 0.0,
        failed_consumption_penalty: float = -0.1,

        # Episode settings
        max_steps: int = 200,

        render_mode: Optional[str] = None
    ):
        """
        Initialize the graph-based empathic environment.

        Args:
            num_agents: Number of agents in the environment
            max_neighbors: Maximum neighbors for observation padding

            initial_pool: Starting resource amount
            max_pool_capacity: Maximum pool capacity
            regeneration_rate: Resource regeneration rate
            regeneration_mode: 'constant', 'proportional', or 'none'

            graph_type: 'random', 'small_world', 'fully_connected', 'ring', 'custom', 'dynamic'
            edge_probability: Edge probability for random graph
            small_world_k: Neighbors per side for small-world
            small_world_p: Rewiring probability for small-world
            custom_adjacency: Custom adjacency matrix
            dynamic_p_base, dynamic_p_max, dynamic_delta, dynamic_decay: For 'dynamic' graph

            memory_size: Agent meal history size
            beta: Satisfaction calculation parameter
            threshold: Emotion threshold
            smoothing: Emotion smoothing type
            sigmoid_gain: Sigmoid gain parameter

            exploit_reward: Base reward for consumption
            empathic_penalty: Penalty parameter for empathic calculation
            empathy_alpha: Weight on others' emotions (0=selfish, 1=fully empathic)
            step_cost: Cost per step
            wait_reward: Reward for waiting
            failed_consumption_penalty: Penalty for failed consumption attempt

            max_steps: Maximum steps per episode
            render_mode: Rendering mode (optional)
        """
        super().__init__()

        self.num_agents = num_agents
        self.max_neighbors = max_neighbors
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Store settings for reset
        self._graph_type = graph_type
        self._edge_probability = edge_probability
        self._small_world_k = small_world_k
        self._small_world_p = small_world_p
        self._custom_adjacency = custom_adjacency

        self._initial_pool = initial_pool
        self._max_pool_capacity = max_pool_capacity
        self._regeneration_rate = regeneration_rate
        self._regeneration_mode = regeneration_mode

        # Agent parameters
        self._memory_size = memory_size
        self._beta = beta
        self._threshold = threshold
        self._smoothing = smoothing
        self._sigmoid_gain = sigmoid_gain

        # Reward settings
        self.exploit_reward = exploit_reward
        self.empathic_penalty = empathic_penalty
        self.empathy_alpha = empathy_alpha
        self.step_cost = step_cost
        self.wait_reward = wait_reward
        self.failed_consumption_penalty = failed_consumption_penalty

        # Empathic reward calculator (from 2D module)
        self._reward_calc = EmpathicRewardCalculator(
            exploit_reward=self.exploit_reward,
            empathic_penalty=self.empathic_penalty,
            empathy_alpha=self.empathy_alpha,
        )

        # Initialize components (will be properly set in reset())
        self.graph: Optional[SocialGraph] = None
        self.pool: Optional[ResourcePool] = None
        self.agents: List = []  # Will be set externally
        self.steps_taken = 0

        # Random number generator
        self.np_random = np.random.default_rng()

        # Observation space: flat vector
        # [own_satisfaction, own_emotion, pool_level, neighbor_emotions_and_weights...]
        obs_dim = 3 + max_neighbors * 2
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: binary (wait=0, consume=1)
        self.action_space = gym.spaces.Discrete(2)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observations, info)
        """
        super().reset(seed=seed)
        self.steps_taken = 0

        # Initialize/reset resource pool
        self.pool = ResourcePool(
            initial_amount=self._initial_pool,
            max_capacity=self._max_pool_capacity,
            regeneration_rate=self._regeneration_rate,
            regeneration_mode=self._regeneration_mode,
            seed=seed
        )

        # Initialize/reset social graph
        self.graph = SocialGraph(
            num_agents=self.num_agents,
            graph_type=self._graph_type,
            edge_probability=self._edge_probability,
            k=self._small_world_k,
            p=self._small_world_p,
            adjacency_matrix=self._custom_adjacency,
            seed=seed,
            p_base=getattr(self, '_dynamic_p_base', 0.05),
            p_max=getattr(self, '_dynamic_p_max', 0.95),
            delta=getattr(self, '_dynamic_delta', 0.02),
            decay=getattr(self, '_dynamic_decay', 0.9),
        )

        # Reset all agents (meal history, satisfaction, etc.)
        for agent in self.agents:
            if hasattr(agent, 'agent'):
                # GraphDQNAgent wraps Agent
                agent.agent.reset()
            elif hasattr(agent, 'reset'):
                agent.reset()

        # Generate initial observations
        observations = [self._get_observation(i) for i in range(self.num_agents)]

        info = {
            'steps': 0,
            'pool_level': self.pool.get_level(),
            'graph_density': self.graph.get_density(),
            'graph_avg_degree': self.graph.get_average_degree(),
        }

        return observations, info

    def step(
        self,
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Execute one step for all agents.

        Args:
            actions: List of actions for each agent (0=wait, 1=consume)

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        self.steps_taken += 1

        rewards = [0.0] * self.num_agents
        consumed_flags = [False] * self.num_agents
        consumption_bonus = [0.0] * self.num_agents
        empathic_penalty_events = [False] * self.num_agents

        # Find agents who want to consume
        consumers = [i for i, action in enumerate(actions) if action == 1]

        # Handle consumption
        if consumers:
            # Try to serve all consumers
            num_served, served_indices = self.pool.consume_multiple(
                num_consumers=len(consumers),
                amount_per_consumer=1
            )

            # Map served_indices back to agent indices
            served_agents = [consumers[i] for i in served_indices]

            # Process each consumer
            for consumer_idx, agent_idx in enumerate(consumers):
                if consumer_idx in served_indices:
                    # Successful consumption
                    consumed_flags[agent_idx] = True

                    # Get graph neighbors for empathic reward
                    neighbor_indices = self.graph.get_neighbors(agent_idx)
                    neighbor_agents = [self.agents[j] for j in neighbor_indices]

                    # Get the actual Agent object (handle GraphDQNAgent wrapper)
                    if hasattr(self.agents[agent_idx], 'agent'):
                        self_agent = self.agents[agent_idx].agent
                        neighbor_agents_unwrapped = [
                            a.agent if hasattr(a, 'agent') else a
                            for a in neighbor_agents
                        ]
                    else:
                        self_agent = self.agents[agent_idx]
                        neighbor_agents_unwrapped = neighbor_agents

                    # Calculate empathic reward
                    bonus = self._reward_calc.consumption_bonus(
                        self_agent=self_agent,
                        nearby_agents=neighbor_agents_unwrapped,
                    )

                    consumption_bonus[agent_idx] = bonus
                    empathic_penalty_events[agent_idx] = bonus < 0.0

                    rewards[agent_idx] = self.step_cost + bonus

                    # Record successful meal
                    self._record_meal(agent_idx, success=True, reward=bonus)
                else:
                    # Failed consumption (pool empty)
                    rewards[agent_idx] = self.step_cost + self.failed_consumption_penalty
                    self._record_meal(agent_idx, success=False, reward=0)

        # Process agents who waited
        for i, action in enumerate(actions):
            if action == 0:  # Wait
                rewards[i] = self.step_cost + self.wait_reward
                self._record_meal(i, success=False, reward=0)

        # Regenerate resources
        self.pool.regenerate()

        # Dynamic graph: evolve connections after this step
        if self._graph_type == 'dynamic' and hasattr(self.graph, 'step_dynamics'):
            self.graph.step_dynamics(self.np_random)

        # Check termination
        terminated = False
        if self._regeneration_mode == 'none' and self.pool.is_empty():
            terminated = True

        truncated = self.steps_taken >= self.max_steps

        # Get new observations
        observations = [self._get_observation(i) for i in range(self.num_agents)]

        # Build info dict
        info = {
            'steps': self.steps_taken,
            'pool_level': self.pool.get_level(),
            'pool_normalized': self.pool.get_normalized_level(),
            'consumed': consumed_flags,
            'consumption_bonus': consumption_bonus,
            'empathic_penalty_events': empathic_penalty_events,
            'satisfactions': [self._get_satisfaction(i) for i in range(self.num_agents)],
            'total_consumption': sum(consumed_flags),
        }

        return observations, rewards, terminated, truncated, info

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Build observation vector for an agent.

        Structure:
        [own_satisfaction, own_emotion, pool_level,
         neighbor_0_emotion, neighbor_0_weight,
         neighbor_1_emotion, neighbor_1_weight, ...]

        Args:
            agent_idx: Index of the agent

        Returns:
            Flat observation array
        """
        obs_dim = 3 + self.max_neighbors * 2
        obs = np.zeros(obs_dim, dtype=np.float32)

        # Own state
        obs[0] = self._get_satisfaction(agent_idx)
        obs[1] = self._get_emotion(agent_idx)
        obs[2] = self.pool.get_normalized_level()

        # Neighbor emotions and weights
        neighbors = self.graph.get_neighbors(agent_idx)
        weights = self.graph.get_neighbor_weights(agent_idx)

        for i, neighbor_idx in enumerate(neighbors[:self.max_neighbors]):
            obs[3 + i * 2] = self._get_emotion(neighbor_idx)
            obs[3 + i * 2 + 1] = weights[neighbor_idx]

        # Remaining slots stay as zeros (padding)

        return obs

    def _get_satisfaction(self, agent_idx: int) -> float:
        """Get satisfaction for an agent (handle wrapper)."""
        agent = self.agents[agent_idx]
        if hasattr(agent, 'get_satisfaction'):
            return agent.get_satisfaction()
        elif hasattr(agent, 'agent'):
            return agent.agent.get_satisfaction()
        return 0.0

    def _get_emotion(self, agent_idx: int) -> float:
        """Get emotion for an agent (handle wrapper)."""
        agent = self.agents[agent_idx]
        if hasattr(agent, 'get_emotion'):
            return agent.get_emotion()
        elif hasattr(agent, 'agent'):
            return agent.agent.get_emotion()
        return 0.0

    def _record_meal(self, agent_idx: int, success: bool, reward: float) -> None:
        """Record meal in agent's history (handle wrapper)."""
        agent = self.agents[agent_idx]
        if hasattr(agent, 'record_meal'):
            agent.record_meal(success, reward)
        elif hasattr(agent, 'agent') and hasattr(agent.agent, 'record_meal'):
            agent.agent.record_meal(success, reward)

    def set_agents(self, agents: List) -> None:
        """
        Link external agent objects to environment.

        Args:
            agents: List of agent objects (GraphDQNAgent or similar)
        """
        if len(agents) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agents, got {len(agents)}"
            )
        self.agents = agents

    def get_graph(self) -> SocialGraph:
        """Return reference to social graph."""
        return self.graph

    def get_pool(self) -> ResourcePool:
        """Return reference to resource pool."""
        return self.pool

    def close(self) -> None:
        """Close the environment."""
        pass

    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass
