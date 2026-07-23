"""
game-theoretic multi-agent environment for the tragedy-of-the-commons scenario.

this module defines GameTheoreticEnv, where N agents share a finite resource pool
and independently decide whether to exploit (action=1) or abstain (action=0) at
each time step.  the resource decreases with exploitation and regenerates at a
configurable rate.

observations can include emotional signals of other agents (average scalar or
full vector), or be zeroed-out when empathy is disabled (see_emotions=False).

reward shaping is handled by SocialRewardCalculator (from agent_policies):
  combined_reward = (1 - alpha) * personal_satisfaction + alpha * empathic_reward

supports both deterministic and stochastic exploitation modes.
"""

import numpy as np
from agent_policies_game_theoretic import QAgent, DQNAgent, SocialRewardCalculator


class GameTheoreticEnv:
    """
    multi-agent environment where each agent can exploit a shared resource or not.
    agents can optionally observe emotional states of others (average or full vector),
    or see nothing.
    """
    def __init__(self, nb_agents,
                 initial_resources=100,
                 regen_rate=1.0,
                 env_type="deterministic",
                 emotion_type="average",
                 see_emotions=True,
                 see_resources=True,
                 alpha=0.5,
                 beta=0.5,
                 agent_class=DQNAgent,
                 agent_configs=None,
                 threshold=0.7,
                 smoothing='linear',
                 sigmoid_gain=10.0,
                 round_emotions=None):

        # environment settings
        self.nb_agents = nb_agents
        self.initial_resources = initial_resources
        self.regen_rate = regen_rate
        self.env_type = env_type              # "deterministic" or "stochastic"
        self.emotion_type = emotion_type      # "average" or "vector" (ignored if see_emotions=False)
        self.see_emotions = see_emotions      # If False, agents receive zero emotion observations
        self.see_resources = see_resources    # If True, append normalized resource level to observations
        self._alpha = alpha                    # empathic weight on others
        self.beta = beta                      # weight of last vs history
        self.n_actions = 2                    # 0: abstain, 1: exploit
        self.actions = np.arange(self.n_actions)
        self.round_emotions = round_emotions

        # agent setup
        self.agent_class = agent_class
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        # social reward calculator
        self.reward_calculator = SocialRewardCalculator(nb_agents,
                                                        alpha=self._alpha,
                                                        beta=beta,
                                                        threshold=threshold,
                                                        smoothing=smoothing,
                                                        sigmoid_gain=sigmoid_gain
                                                        )

        # initialize agents and reset environment state
        self._init_agents()
        self.reset()

        self._alpha_initial = alpha

    @property
    def alpha(self):
        return self._alpha

    def _init_agents(self):
        # determine input dimension for agents based on emotion visibility
        if not self.see_emotions:
            self.state_size = 1
        else:
            if self.emotion_type == "average":
                self.state_size = 1
            elif self.emotion_type == "vector":
                self.state_size = self.nb_agents
            else:
                raise ValueError(f"Unknown emotion_type: {self.emotion_type}")

        # resource visibility adds one extra observation dimension (orthogonal to emotions)
        if self.see_resources:
            self.state_size += 1

        # instantiate agent objects with per-agent configs
        self.agents = []
        for idx, config in enumerate(self.agent_configs):
            agent = self.agent_class(
                state_size=self.state_size,
                action_size=self.n_actions,
                agent_id=idx,
                **config
            )
            self.agents.append(agent)

    def reset(self):
        self.resource = self.initial_resources
        self.time_step = 0

        obs = self.get_observations()

        for idx, agent in enumerate(self.agents):
            agent.reset(observation=obs[idx])

            if hasattr(agent, 'memory'):
                agent.memory.buffer.clear()

        return obs

    def _resource_signal(self):
        """
        return the shared resource level normalized to [0, 1]
        (resource / initial_resources), rounded like the emotion signals.
        this is a global scalar, identical for every agent.
        """
        signal = self.resource / self.initial_resources if self.initial_resources else 0.0
        signal = float(np.clip(signal, 0.0, 1.0))
        if self.round_emotions is not None:
            signal = round(signal, self.round_emotions)
        return signal

    def get_observations(self):
        """
        return list of observations for each agent.

        the emotion part is unchanged (others’ emotions as average or vector, or a
        zero placeholder when see_emotions=False).  when see_resources=True, the
        normalized shared-resource level is appended as an extra dimension.
        """

        # emotion part of the observation (unchanged behaviour)
        if not self.see_emotions:
            # zero placeholder, same size as the original emotion observation
            emotion_dim = self.state_size - (1 if self.see_resources else 0)
            observations = [np.zeros(emotion_dim, dtype=float)
                            for _ in range(self.nb_agents)]
        else:
            emotions = self.reward_calculator.calculate_emotions(self.agents)

            if self.round_emotions is not None:
                emotions = np.round(emotions, self.round_emotions)

            observations = []
            for i in range(self.nb_agents):
                other_emotions = [e for j, e in enumerate(emotions) if j != i]
                if self.emotion_type == "average":
                    obs = np.array([np.mean(other_emotions)], dtype=float)
                elif self.emotion_type == "vector":
                    obs = np.array(other_emotions, dtype=float)
                else:
                    raise ValueError(f"Unknown emotion_type: {self.emotion_type}")
                observations.append(obs)

        # append the global resource signal as an extra dimension (orthogonal axis)
        if self.see_resources:
            resource_signal = self._resource_signal()
            observations = [np.append(obs, resource_signal) for obs in observations]

        return observations

    def step(self, actions):
        """
        execute one timestep: agents choose actions, environment updates.
        actions: list of 0/1 for each agent.
        returns: next_observations, rewards, done, info.
        """
        consumed = 0
        immediate_rewards = []

        for idx, act in enumerate(actions):
            reward = 0.0
            success = False
            if act == 1 and self.resource > 0:
                if self.env_type == "stochastic":
                    # success probability decreases as resources are depleted
                    prob = self.resource / self.initial_resources
                    success = np.random.rand() < prob
                else:
                    # deterministic: exploitation always succeeds if resources remain
                    success = True
                if success:
                    reward = 1.0
                    consumed += 1

            if hasattr(self.agents[idx], 'record_meal'):
                self.agents[idx].record_meal(success, reward)
            else:
                raise ValueError('problem in the record meal')
            immediate_rewards.append(reward)

        emotions, personal_reward, empathic_reward, total_reward = self.reward_calculator.calculate_rewards(self.agents)

        # update resource pool and step counter
        self.resource = max(0.0, (self.resource - consumed) * self.regen_rate)
        self.time_step += 1
        next_obs = self.get_observations()
        # episode ends when the resource pool is fully depleted
        done = self.resource <= 0

        info = {
            'emotions': emotions,
            'personal_reward': personal_reward,
            'empathic_reward': empathic_reward,
            'combined_reward': total_reward
        }

        return next_obs, total_reward, done, info

    def get_agent_meal_stats(self, agent_idx):
        """return recent and total meals for one agent."""
        a = self.agents[agent_idx]
        return {
            'recent_meals': a.get_recent_meals(),
            'total_meals': a.total_meals,
            'meal_history': list(a.meal_history)
        }

    def get_all_agents_meal_stats(self):
        """return meal stats for all agents."""
        return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
