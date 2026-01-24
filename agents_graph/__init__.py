"""
Graph-Based Multi-Agent Empathic Resource Sharing Environment

This module implements a graph-based variant where:
- Agents are fixed nodes in a social graph (no movement)
- Graph edges determine who can see whose emotions
- Agents share a centralized resource pool
- Static topology with hooks for future affiliation theory
"""

from .config import *
from .social_graph import SocialGraph
from .resource_pool import ResourcePool
from .agent_graph import GraphDQNAgent
from .env_graph import EmpathicGraphEnv

__all__ = [
    'SocialGraph',
    'ResourcePool',
    'GraphDQNAgent',
    'EmpathicGraphEnv',
]
