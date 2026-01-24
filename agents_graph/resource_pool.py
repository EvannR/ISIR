"""
Resource Pool Module

Centralized resource pool shared by all agents.
Supports configurable regeneration modes.
"""

import numpy as np
from typing import Optional, List, Tuple


class ResourcePool:
    """
    Centralized resource pool that all agents share.

    Resources can be consumed by agents and regenerate based on
    configurable modes (constant rate, proportional, or no regeneration).

    Attributes:
        current: Current resource amount
        max_capacity: Maximum pool capacity
        initial_amount: Amount at episode start
        regeneration_rate: Base regeneration rate
        regeneration_mode: How resources regenerate
    """

    def __init__(
        self,
        initial_amount: int = 50,
        max_capacity: int = 100,
        regeneration_rate: float = 1.0,
        regeneration_mode: str = 'constant',
        seed: Optional[int] = None
    ):
        """
        Initialize the resource pool.

        Args:
            initial_amount: Starting resource amount
            max_capacity: Maximum pool size
            regeneration_rate: Base rate for regeneration
            regeneration_mode: One of:
                - 'constant': Fixed regeneration per step
                - 'proportional': Rate scales with current/max ratio
                - 'none': No regeneration
            seed: Random seed for any stochastic behavior
        """
        self.initial_amount = initial_amount
        self.max_capacity = max_capacity
        self.regeneration_rate = regeneration_rate
        self.regeneration_mode = regeneration_mode
        self.rng = np.random.default_rng(seed)

        self.current = float(initial_amount)
        self._total_consumed = 0
        self._total_regenerated = 0.0

    def consume(self, amount: int = 1) -> bool:
        """
        Attempt to consume resources from the pool.

        Args:
            amount: Amount to consume

        Returns:
            True if consumption succeeded, False if pool has insufficient resources
        """
        if self.current >= amount:
            self.current -= amount
            self._total_consumed += amount
            return True
        return False

    def consume_multiple(
        self,
        num_consumers: int,
        amount_per_consumer: int = 1
    ) -> Tuple[int, List[int]]:
        """
        Handle multiple simultaneous consumption requests.

        If pool has enough resources for all consumers, all get served.
        If pool is scarce, randomly select which consumers get served.

        Args:
            num_consumers: Number of agents trying to consume
            amount_per_consumer: Amount each consumer wants

        Returns:
            Tuple of (successful_count, list_of_indices_served)
            The indices are 0 to num_consumers-1, representing which
            consumers were served.
        """
        total_requested = num_consumers * amount_per_consumer
        available = int(self.current)

        if total_requested <= available:
            # All consumers can be served
            self.current -= total_requested
            self._total_consumed += total_requested
            return num_consumers, list(range(num_consumers))

        # Not enough for everyone - randomly select who gets served
        max_served = available // amount_per_consumer
        if max_served == 0:
            return 0, []

        # Random selection
        served_indices = self.rng.choice(
            num_consumers,
            size=max_served,
            replace=False
        ).tolist()
        served_indices.sort()  # Keep order for consistency

        actual_consumed = max_served * amount_per_consumer
        self.current -= actual_consumed
        self._total_consumed += actual_consumed

        return max_served, served_indices

    def regenerate(self) -> float:
        """
        Regenerate resources based on mode.

        Called once per step by the environment.

        Returns:
            Amount regenerated
        """
        if self.regeneration_mode == 'none':
            return 0.0

        old_level = self.current

        if self.regeneration_mode == 'constant':
            # Fixed amount per step
            self.current = min(
                self.current + self.regeneration_rate,
                self.max_capacity
            )
        elif self.regeneration_mode == 'proportional':
            # Rate scales with current level (more resources = faster regen)
            rate = self.regeneration_rate * (self.current / self.max_capacity)
            self.current = min(self.current + rate, self.max_capacity)

        regenerated = self.current - old_level
        self._total_regenerated += regenerated
        return regenerated

    def get_level(self) -> int:
        """Return current resource amount (integer)."""
        return int(self.current)

    def get_normalized_level(self) -> float:
        """Return current level as fraction of max capacity [0, 1]."""
        return self.current / self.max_capacity if self.max_capacity > 0 else 0.0

    def is_empty(self) -> bool:
        """Check if pool is depleted."""
        return self.current < 1

    def is_full(self) -> bool:
        """Check if pool is at max capacity."""
        return self.current >= self.max_capacity

    def reset(self, amount: Optional[int] = None) -> None:
        """
        Reset pool for new episode.

        Args:
            amount: Starting amount (uses initial_amount if not specified)
        """
        self.current = float(amount if amount is not None else self.initial_amount)
        self._total_consumed = 0
        self._total_regenerated = 0.0

    def get_stats(self) -> dict:
        """
        Return statistics about pool usage.

        Returns:
            Dictionary with current, consumed, regenerated, etc.
        """
        return {
            'current': self.get_level(),
            'normalized_level': self.get_normalized_level(),
            'total_consumed': self._total_consumed,
            'total_regenerated': self._total_regenerated,
            'max_capacity': self.max_capacity,
            'is_empty': self.is_empty(),
        }

    def __repr__(self) -> str:
        return (
            f"ResourcePool(current={self.get_level()}/{self.max_capacity}, "
            f"mode={self.regeneration_mode})"
        )
