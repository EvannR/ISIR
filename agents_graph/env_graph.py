"""
Graph-Based Empathic Environment

Multi-agent environment where:
- Agents are fixed nodes in a static directed social graph
- Graph edges determine who can perceive whose emotions
- Agents share a centralized resource pool
- Actions: wait (0) or consume (1)
- Reward follows game theoretic formula:
  consuming agents get (1-alpha)*personal + alpha*empathic
  waiting/failed agents get 0.0
"""

import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Optional, Any

try:
    from .social_graph import SocialGraph
    from .resource_pool import ResourcePool
except ImportError:
    from social_graph import SocialGraph
    from resource_pool import ResourcePool


class EmpathicGraphEnv(gym.Env):
    """
    Multi-agent graph-based environment for empathic resource sharing.

    Key features:
    - Agents are fixed nodes in a static directed social graph
    - Graph edges define emotional visibility (A->B means A perceives B's emotion)
    - Shared resource pool
    - Binary actions: wait (0) or consume (1)
    - Reward: consuming agents get (1-alpha)*personal + alpha*mean_neighbor_emotion
    """

    def __init__(
        self,
        num_agents: int = 6,

        # Resource pool settings
        initial_pool: int = 50,
        max_pool_capacity: int = 100,
        regeneration_rate: float = 1.0,
        regeneration_mode: str = 'constant',

        # Graph settings
        graph_type: str = 'directed_random',
        edge_probability: float = 0.4,
        small_world_k: int = 2,
        small_world_p: float = 0.3,
        custom_adjacency: Optional[np.ndarray] = None,

        # Agent settings
        memory_size: int = 10,
        beta: float = 0.5,
        threshold: float = 0.5,
        smoothing: str = 'linear',
        sigmoid_gain: float = 10.0,

        # Reward settings
        alpha: float = 0.5,

        # Environment type
        env_type: str = 'deterministic',

        # Episode settings
        max_steps: int = 200,
        static_graph_across_episodes: bool = False,

        render_mode: Optional[str] = None
    ):
        """
        Initialize the graph-based empathic environment.

        Args:
            num_agents: Number of agents in the environment

            initial_pool: Starting resource amount
            max_pool_capacity: Maximum pool capacity
            regeneration_rate: Resource regeneration rate
            regeneration_mode: 'constant', 'proportional', or 'none'

            graph_type: 'directed_random', 'random', 'small_world',
                       'fully_connected', 'ring', 'custom'
            edge_probability: Edge probability for random graphs
            small_world_k: Neighbors per side for small-world
            small_world_p: Rewiring probability for small-world
            custom_adjacency: Custom adjacency matrix

            memory_size: Agent meal history size
            beta: Weight of last meal vs history in personal satisfaction
            threshold: Emotion threshold
            smoothing: Emotion smoothing type ('linear' or 'sigmoid')
            sigmoid_gain: Sigmoid gain parameter

            alpha: Empathy weight (0=selfish, 1=fully empathic)

            max_steps: Maximum steps per episode
            static_graph_across_episodes: If True, graph is created once and
                reused across all episodes. If False (default), a new graph
                is generated each reset().
            render_mode: Rendering mode (optional)
        """
        super().__init__()

        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.alpha = alpha
        self._beta = beta
        self.env_type = env_type
        self._static_graph_across_episodes = static_graph_across_episodes

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
        self._threshold = threshold
        self._smoothing = smoothing
        self._sigmoid_gain = sigmoid_gain

        # Initialize components (will be properly set in reset())
        self.graph: Optional[SocialGraph] = None
        self.pool: Optional[ResourcePool] = None
        self.agents: List = []
        self.steps_taken = 0

        self.np_random = np.random.default_rng()

        # Observation space: [own_satisfaction, own_emotion, pool_level, mean_neighbor_emotion]
        obs_dim = 4
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

        # Initialize social graph (optionally reuse across episodes)
        if self.graph is None or not self._static_graph_across_episodes:
            self.graph = SocialGraph(
                num_agents=self.num_agents,
                graph_type=self._graph_type,
                edge_probability=self._edge_probability,
                k=self._small_world_k,
                p=self._small_world_p,
                adjacency_matrix=self._custom_adjacency,
                seed=seed,
            )

        # Reset all agents
        for agent in self.agents:
            if hasattr(agent, 'agent'):
                agent.agent.reset()
            elif hasattr(agent, 'reset'):
                agent.reset()

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

        Reward logic:
        - Consuming agents (successful): (1-alpha)*personal + alpha*empathic
        - Waiting agents: 0.0
        - Failed consumption (pool empty): 0.0

        Args:
            actions: List of actions for each agent (0=wait, 1=consume)

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        self.steps_taken += 1

        rewards = [0.0] * self.num_agents
        consumed_flags = [False] * self.num_agents
        personal_rewards = [0.0] * self.num_agents
        empathic_rewards = [0.0] * self.num_agents

        # Find agents who want to consume
        consumers = [i for i, action in enumerate(actions) if action == 1]

        # Handle consumption (stochastic or deterministic)
        consumed_count = 0
        for agent_idx in consumers:
            if self.pool.current < 1:
                # Pool empty, consumption fails
                self._record_meal(agent_idx, success=False, reward=0)
                continue

            if self.env_type == 'stochastic':
                # Success probability depends on resource level (like game theoretic env)
                prob = self.pool.current / self._initial_pool
                success = self.np_random.random() < prob
            else:
                success = True

            if success:
                self.pool.current -= 1
                consumed_count += 1
                consumed_flags[agent_idx] = True

                # Calculate game theoretic reward
                personal = self._get_personal_satisfaction(agent_idx)
                empathic = self._get_mean_neighbor_emotion(agent_idx)

                personal_rewards[agent_idx] = personal
                empathic_rewards[agent_idx] = empathic
                rewards[agent_idx] = (1 - self.alpha) * personal + self.alpha * empathic

                self._record_meal(agent_idx, success=True, reward=rewards[agent_idx])
            else:
                # Stochastic failure
                self._record_meal(agent_idx, success=False, reward=0)

        # Record non-consumption for waiting agents
        for i, action in enumerate(actions):
            if action == 0:
                self._record_meal(i, success=False, reward=0)

        # Multiplicative resource update (like game theoretic env)
        self.pool.current = max(0.0, self.pool.current * self._regeneration_rate)

        # Check termination — episode ends when pool is depleted
        terminated = self.pool.current <= 0

        truncated = self.steps_taken >= self.max_steps

        observations = [self._get_observation(i) for i in range(self.num_agents)]

        info = {
            'steps': self.steps_taken,
            'pool_level': self.pool.get_level(),
            'pool_normalized': self.pool.get_normalized_level(),
            'consumed': consumed_flags,
            'personal_rewards': personal_rewards,
            'empathic_rewards': empathic_rewards,
            'satisfactions': [self._get_satisfaction(i) for i in range(self.num_agents)],
            'emotions': [self._get_emotion(i) for i in range(self.num_agents)],
            'total_consumption': sum(consumed_flags),
        }

        return observations, rewards, terminated, truncated, info

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Build observation vector for an agent.

        Structure: [own_satisfaction, own_emotion, pool_level, mean_neighbor_emotion]

        Args:
            agent_idx: Index of the agent

        Returns:
            Flat observation array of dim 4
        """
        obs = np.zeros(4, dtype=np.float32)

        obs[0] = self._get_satisfaction(agent_idx)
        obs[1] = self._get_emotion(agent_idx)
        obs[2] = self.pool.get_normalized_level()
        obs[3] = self._get_mean_neighbor_emotion(agent_idx)

        return obs

    def _get_personal_satisfaction(self, agent_idx: int) -> float:
        """
        Calculate personal satisfaction (game theoretic formula).
        personal = beta * last_meal + (1-beta) * avg_history
        """
        agent = self.agents[agent_idx]
        if hasattr(agent, 'agent'):
            agent = agent.agent

        if hasattr(agent, 'meal_history') and agent.meal_history:
            last = 1.0 if agent.meal_history[-1] > 0 else 0.0
            hist = sum(agent.meal_history) / len(agent.meal_history)
        else:
            last, hist = 0.0, 0.0

        return self._beta * last + (1 - self._beta) * hist

    def _get_mean_neighbor_emotion(self, agent_idx: int) -> float:
        """
        Get mean emotion of agents this agent is connected to (outgoing edges).
        """
        neighbors = self.graph.get_neighbors(agent_idx)
        if not neighbors:
            return 0.0
        emotions = [self._get_emotion(j) for j in neighbors]
        return float(np.mean(emotions))

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
        """Link external agent objects to environment."""
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
        pass

    def render(self) -> None:
        pass
