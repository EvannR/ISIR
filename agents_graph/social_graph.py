"""
Social Graph Module

Manages social graph topology and neighbor relationships.
Edges define who can "see" whose emotions for empathic reward calculation.
Supports static topology with weight hooks for future affiliation theory.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class SocialGraph:
    """
    Manages social connections between agents as a graph structure.

    Attributes:
        num_agents: Number of agents (nodes) in the graph
        adjacency: Binary adjacency matrix (num_agents x num_agents)
        weights: Edge weights matrix for future affiliation theory
        directed: Whether graph is directed (default: False, undirected)
    """

    def __init__(
        self,
        num_agents: int,
        graph_type: str = 'random',
        edge_probability: float = 0.3,
        k: int = 2,                          # Small-world: neighbors per side
        p: float = 0.1,                      # Small-world: rewiring probability
        adjacency_matrix: Optional[np.ndarray] = None,
        directed: bool = False,
        seed: Optional[int] = None,
        # Dynamic graph parameters (only used when graph_type == 'dynamic')
        p_base: float = 0.05,
        p_max: float = 0.95,
        delta: float = 0.02,
        decay: float = 0.9,
    ):
        """
        Initialize the social graph.

        Args:
            num_agents: Number of agents (nodes)
            graph_type: Type of graph - 'random', 'small_world', 'fully_connected',
                       'ring', 'custom', or 'dynamic'
            edge_probability: Edge probability for random graph (Erdos-Renyi)
            k: Number of neighbors on each side for small-world (Watts-Strogatz)
            p: Rewiring probability for small-world
            adjacency_matrix: Custom adjacency matrix (for 'custom' type)
            directed: If True, edges are directed; if False, symmetric
            seed: Random seed for reproducibility
            p_base: (dynamic) Base probability to connect each step
            p_max: (dynamic) Max connection probability after reinforcement
            delta: (dynamic) Increase in p when edge is present
            decay: (dynamic) Decay toward p_base when edge absent
        """
        self.num_agents = num_agents
        self.directed = directed
        self.rng = np.random.default_rng(seed)

        # Initialize adjacency and weights matrices
        self.adjacency = np.zeros((num_agents, num_agents), dtype=np.float32)
        self.weights = np.ones((num_agents, num_agents), dtype=np.float32)

        # Build the graph based on type
        self._build_graph(
            graph_type, edge_probability, k, p, adjacency_matrix,
            p_base=p_base, p_max=p_max, delta=delta, decay=decay,
        )

        # Zero out diagonal (no self-loops)
        np.fill_diagonal(self.adjacency, 0)
        np.fill_diagonal(self.weights, 0)

    def _build_graph(
        self,
        graph_type: str,
        edge_probability: float,
        k: int,
        p: float,
        adjacency_matrix: Optional[np.ndarray],
        p_base: float = 0.05,
        p_max: float = 0.95,
        delta: float = 0.02,
        decay: float = 0.9,
    ):
        """Build graph based on specified type."""
        if graph_type == 'random':
            self._build_random(edge_probability)
        elif graph_type == 'small_world':
            self._build_small_world(k, p)
        elif graph_type == 'fully_connected':
            self._build_fully_connected()
        elif graph_type == 'ring':
            self._build_ring()
        elif graph_type == 'custom':
            self._load_custom(adjacency_matrix)
        elif graph_type == 'directed_random':
            self.directed = True
            self._build_directed_random(edge_probability)
        elif graph_type == 'dynamic':
            self._build_dynamic(p_base, p_max, delta, decay)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    def _build_random(self, edge_probability: float):
        """
        Build Erdos-Renyi random graph.
        Each edge exists with probability edge_probability.
        """
        n = self.num_agents
        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < edge_probability:
                    self.adjacency[i, j] = 1.0
                    if not self.directed:
                        self.adjacency[j, i] = 1.0

    def _build_small_world(self, k: int, p: float):
        """
        Build Watts-Strogatz small-world graph.

        1. Start with a ring lattice where each node connects to k neighbors on each side
        2. Rewire each edge with probability p
        """
        n = self.num_agents

        # Step 1: Create ring lattice
        for i in range(n):
            for j in range(1, k + 1):
                # Connect to k neighbors on each side
                right = (i + j) % n
                left = (i - j) % n
                self.adjacency[i, right] = 1.0
                self.adjacency[i, left] = 1.0
                if not self.directed:
                    self.adjacency[right, i] = 1.0
                    self.adjacency[left, i] = 1.0

        # Step 2: Rewire edges with probability p
        for i in range(n):
            for j in range(1, k + 1):
                if self.rng.random() < p:
                    # Remove current edge
                    old_target = (i + j) % n
                    self.adjacency[i, old_target] = 0.0
                    if not self.directed:
                        self.adjacency[old_target, i] = 0.0

                    # Find new random target (not self, not already connected)
                    candidates = [
                        x for x in range(n)
                        if x != i and self.adjacency[i, x] == 0
                    ]
                    if candidates:
                        new_target = self.rng.choice(candidates)
                        self.adjacency[i, new_target] = 1.0
                        if not self.directed:
                            self.adjacency[new_target, i] = 1.0

    def _build_directed_random(self, edge_probability: float):
        """
        Build random directed graph (Erdos-Renyi style, asymmetric).

        Each ordered pair (i,j) has independent probability of edge i->j.
        i->j means agent i perceives agent j's emotion.
        Guarantees every agent has at least one outgoing edge.
        """
        n = self.num_agents
        for i in range(n):
            for j in range(n):
                if i != j and self.rng.random() < edge_probability:
                    self.adjacency[i, j] = 1.0

        # Guarantee: every agent has at least one outgoing edge
        for i in range(n):
            if np.sum(self.adjacency[i]) == 0:
                # Add one random outgoing edge
                candidates = [j for j in range(n) if j != i]
                target = self.rng.choice(candidates)
                self.adjacency[i, target] = 1.0

    def _build_fully_connected(self):
        """Build fully connected graph (complete graph)."""
        self.adjacency = np.ones((self.num_agents, self.num_agents), dtype=np.float32)
        np.fill_diagonal(self.adjacency, 0)

    def _build_ring(self):
        """Build simple ring graph (each node connects to immediate neighbors)."""
        n = self.num_agents
        for i in range(n):
            next_node = (i + 1) % n
            self.adjacency[i, next_node] = 1.0
            if not self.directed:
                self.adjacency[next_node, i] = 1.0

    def _load_custom(self, adjacency_matrix: Optional[np.ndarray]):
        """Load custom adjacency matrix."""
        if adjacency_matrix is None:
            raise ValueError("adjacency_matrix required for 'custom' graph type")

        if adjacency_matrix.shape != (self.num_agents, self.num_agents):
            raise ValueError(
                f"adjacency_matrix shape {adjacency_matrix.shape} doesn't match "
                f"num_agents ({self.num_agents})"
            )

        self.adjacency = adjacency_matrix.astype(np.float32).copy()

    def _build_dynamic(
        self,
        p_base: float,
        p_max: float,
        delta: float,
        decay: float,
    ) -> None:
        """
        Build dynamic graph: start with no edges.
        Per-pair connection probability p_ij is stored and updated each step via step_dynamics().
        """
        n = self.num_agents
        # Adjacency stays zeros (no edges initially)
        self._connection_prob = np.full((n, n), p_base, dtype=np.float32)
        np.fill_diagonal(self._connection_prob, 0)
        self._dynamic_p_base = p_base
        self._dynamic_p_max = p_max
        self._dynamic_delta = delta
        self._dynamic_decay = decay

    def step_dynamics(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Update graph for dynamic type: each pair (i,j) connects with probability p_ij.
        If connected this step: p_ij increases (reinforcement). If not: p_ij decays toward p_base.
        Edges can disappear; past connections make future connection more likely.
        """
        if not hasattr(self, '_connection_prob'):
            return
        rng = rng if rng is not None else self.rng
        n = self.num_agents
        p_base = self._dynamic_p_base
        p_max = self._dynamic_p_max
        delta = self._dynamic_delta
        decay = self._dynamic_decay

        # New adjacency: for each pair, connect with probability p_ij
        new_adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < self._connection_prob[i, j]:
                    new_adj[i, j] = 1.0
                    new_adj[j, i] = 1.0

        # Update connection probabilities
        for i in range(n):
            for j in range(i + 1, n):
                if new_adj[i, j] > 0:
                    # Reinforcement
                    self._connection_prob[i, j] = min(
                        p_max,
                        self._connection_prob[i, j] + delta,
                    )
                    self._connection_prob[j, i] = self._connection_prob[i, j]
                else:
                    # Decay toward p_base
                    self._connection_prob[i, j] = (
                        decay * self._connection_prob[i, j] + (1.0 - decay) * p_base
                    )
                    self._connection_prob[j, i] = self._connection_prob[i, j]

        self.adjacency = new_adj
        np.fill_diagonal(self.adjacency, 0)

    def get_neighbors(self, agent_idx: int) -> List[int]:
        """
        Return list of neighbor indices for an agent.

        Args:
            agent_idx: Index of the agent

        Returns:
            List of indices of connected agents
        """
        return np.where(self.adjacency[agent_idx] > 0)[0].tolist()

    def get_neighbor_weights(self, agent_idx: int) -> Dict[int, float]:
        """
        Return {neighbor_idx: weight} dict for an agent.

        Args:
            agent_idx: Index of the agent

        Returns:
            Dictionary mapping neighbor indices to edge weights
        """
        neighbors = self.get_neighbors(agent_idx)
        return {n: float(self.weights[agent_idx, n]) for n in neighbors}

    def get_degree(self, agent_idx: int) -> int:
        """Return number of connections (degree) for an agent."""
        return int(np.sum(self.adjacency[agent_idx] > 0))

    def get_all_degrees(self) -> List[int]:
        """Return list of degrees for all agents."""
        return [self.get_degree(i) for i in range(self.num_agents)]

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return copy of adjacency matrix."""
        return self.adjacency.copy()

    def get_weights_matrix(self) -> np.ndarray:
        """Return copy of weights matrix."""
        return self.weights.copy()

    def is_connected(self, agent_i: int, agent_j: int) -> bool:
        """Check if two agents are connected."""
        return self.adjacency[agent_i, agent_j] > 0

    def get_edge_weight(self, agent_i: int, agent_j: int) -> float:
        """Get weight of edge between two agents (0 if not connected)."""
        if self.adjacency[agent_i, agent_j] > 0:
            return float(self.weights[agent_i, agent_j])
        return 0.0

    # ----------------------------------------
    # Affiliation Theory Hooks (Future Use)
    # ----------------------------------------

    def update_weight(
        self,
        agent_i: int,
        agent_j: int,
        interaction_type: str,
        strengthen_factor: float = 1.1,
        weaken_factor: float = 0.95,
        min_weight: float = 0.1,
        max_weight: float = 2.0
    ) -> None:
        """
        Update edge weight based on interaction type.

        Hook for future affiliation theory implementation:
        - Cooperative interactions strengthen bonds
        - Competitive interactions weaken bonds

        Args:
            agent_i: First agent index
            agent_j: Second agent index
            interaction_type: 'cooperative' or 'competitive'
            strengthen_factor: Multiplicative factor for strengthening
            weaken_factor: Multiplicative factor for weakening
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        if not self.is_connected(agent_i, agent_j):
            return

        if interaction_type == 'cooperative':
            self.weights[agent_i, agent_j] *= strengthen_factor
            if not self.directed:
                self.weights[agent_j, agent_i] *= strengthen_factor
        elif interaction_type == 'competitive':
            self.weights[agent_i, agent_j] *= weaken_factor
            if not self.directed:
                self.weights[agent_j, agent_i] *= weaken_factor

        # Clamp weights
        self.weights = np.clip(self.weights, min_weight, max_weight)

    def reset_weights(self) -> None:
        """Reset all edge weights to 1.0."""
        self.weights = np.ones_like(self.weights)
        np.fill_diagonal(self.weights, 0)

    # ----------------------------------------
    # Statistics and Visualization
    # ----------------------------------------

    def get_density(self) -> float:
        """
        Calculate graph density (ratio of actual edges to possible edges).

        Returns:
            Density value between 0 and 1
        """
        n = self.num_agents
        max_edges = n * (n - 1) if self.directed else n * (n - 1) / 2
        actual_edges = np.sum(self.adjacency > 0)
        if not self.directed:
            actual_edges /= 2
        return actual_edges / max_edges if max_edges > 0 else 0.0

    def get_average_degree(self) -> float:
        """Calculate average node degree."""
        degrees = self.get_all_degrees()
        return np.mean(degrees) if degrees else 0.0

    def draw_graph(self, ax=None, title=None):
        """
        Visualize the graph using networkx + matplotlib.

        For directed graphs, arrows show perception direction:
        A -> B means A perceives B's emotion.

        Args:
            ax: matplotlib Axes to draw on (creates new figure if None)
            title: optional title for the plot

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig = ax.get_figure()

        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(range(self.num_agents))

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if self.adjacency[i, j] > 0:
                    G.add_edge(i, j)

        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                               node_color='lightblue', edgecolors='black')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=self.directed,
                               arrowsize=20, connectionstyle='arc3,rad=0.1',
                               edge_color='gray', width=1.5)

        if title:
            ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        return fig

    def __repr__(self) -> str:
        return (
            f"SocialGraph(num_agents={self.num_agents}, "
            f"density={self.get_density():.2f}, "
            f"avg_degree={self.get_average_degree():.1f})"
        )
