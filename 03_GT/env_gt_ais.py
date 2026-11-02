"""
game-theoretic multi-agent environment for the AIS-2026 revision experiments (02_GT_AIS).

derived from empathy_tragedy/game_theoretic_matrix_setup/env_game_theoretic.py with
the changes requested by the AIS 2026 reviews:

  1. OBSERVATION (answer to R1's "extreme sensory bottleneck"):
     every agent now receives a fixed-size observation vector
         [ resource_slot, own_meals_slot, social_slot(s), (time_slot) ]
     where
       - resource_slot  = r_t / r_0 if agent.see_resource else 0.0
       - own_meals_slot = sum(meal_history) / memory_size if agent.see_own_meals
                          else 0.0 — the agent's own recent consumption rate,
                          i.e. the internal state its personal satisfaction and
                          broadcast valence are computed from (fixes the
                          reward-not-Markovian-in-observation issue)
       - social_slot    = mean (or vector) of others' emotions if agent.see_emotions else 0.0
       - time_slot      = t / max_steps, only present if include_time=True
     the dimensionality is IDENTICAL across all conditions — information is
     gated to zero, the architecture never changes. this keeps the network
     comparable between conditions while removing the sensory bottleneck.

  2. REPLAY BUFFER: reset() no longer clears agents' replay buffers. experience
     persists across episodes, as in standard DQN.

  3. REWARD MODE: reward_mode='social' uses the emotion-based
     SocialRewardCalculator (manuscript formula). reward_mode='inequity_aversion'
     uses InequityAversionCalculator (Hughes et al. 2018 baseline). emotions are
     always computed by the social calculator so the observation channel is
     identical in both modes.

  4. info now also exposes 'immediate_rewards' (per-agent 0/1 meal outcomes of
     the step) so the runner can log per-episode consumption and compute
     inequality metrics (Gini).

03_GT (journal version):

  5. GRADED SOCIAL OBSERVATION: the social slot(s) are no longer either 0.0 or
     the exact emotion. each agent perceives them through perceive_bounded()
     with its own emotion_precision rho in [0,1] (rho = 0: uniform noise;
     rho = 1: exact value). see_emotions=False now means rho = 0, i.e. uniform
     noise instead of a constant 0.0. the social slot is expressed on the
     reward scheme's emotion scale ([0,1] homeostatic, [-1,1] legacy), so what
     the agent observes is the quantity its empathic reward is computed from.
  6. reward_scheme ('homeostatic', 'homeostatic_restraint' or 'legacy_ais2026')
     is forwarded to SocialRewardCalculator — see agent_policies_gt_ais. step()
     passes the actions to the calculator ('homeostatic_restraint' rewards the
     choice to abstain, not a failed meal).
"""

import numpy as np
from agent_policies_gt_ais import (QAgent, DQNAgent, FixedPolicyAgent,
                                   SocialRewardCalculator,
                                   InequityAversionCalculator,
                                   perceive_bounded)


class GameTheoreticEnv:
    """
    multi-agent environment where each agent can exploit a shared resource or not.

    each agent has its own see_resource, see_emotions and alpha attributes
    (set via agent_configs), enabling the factorial designs of the revision:
      - see_resource=False/True : resource-observation dimension (NEW)
      - see_emotions=False/True : social-observation dimension
      - alpha=0.0/0.5           : reward dimension
    """
    def __init__(self, nb_agents,
                 initial_resources=100,
                 regen_rate=1.0,
                 env_type="deterministic",
                 emotion_type="average",
                 see_emotions=True,      # env-level default (overridden by per-agent config)
                 see_resource=True,      # env-level default (overridden by per-agent config)
                 see_own_meals=True,     # env-level default (overridden by per-agent config)
                 alpha=0.5,              # env-level default (overridden by per-agent config)
                 emotion_precision=None, # env-level default (overridden by per-agent config)
                 beta=0.5,
                 agent_class=DQNAgent,
                 agent_configs=None,
                 threshold=0.7,
                 smoothing='linear',
                 sigmoid_gain=10.0,
                 round_emotions=None,
                 include_time=False,
                 max_steps=1000,
                 reward_mode="social",
                 ia_params=None,
                 reward_scheme="homeostatic"):

        # environment settings
        self.nb_agents = nb_agents
        self.initial_resources = initial_resources
        self.regen_rate = regen_rate
        self.env_type = env_type              # "deterministic" or "stochastic"
        self.emotion_type = emotion_type      # "average" or "vector"
        self.see_emotions = see_emotions      # env-level default (used as fallback in _init_agents)
        self.see_resource = see_resource      # env-level default (used as fallback in _init_agents)
        self.see_own_meals = see_own_meals    # env-level default (used as fallback in _init_agents)
        self._alpha = alpha                   # env-level default (used as fallback in _init_agents)
        self.emotion_precision = emotion_precision  # env-level default (used as fallback in _init_agents)
        self.beta = beta                      # weight of last vs history
        self.n_actions = 2                    # 0: abstain, 1: exploit
        self.actions = np.arange(self.n_actions)
        self.round_emotions = round_emotions
        self._threshold = threshold
        self._smoothing = smoothing
        self._sigmoid_gain = sigmoid_gain
        self.include_time = include_time
        self.max_steps = max_steps

        if reward_mode not in ("social", "inequity_aversion"):
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        self.reward_mode = reward_mode

        # agent setup
        self.agent_class = agent_class
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        # init agents FIRST so per-agent alphas can be read for the calculators
        self._init_agents()

        # emotion/social calculator is ALWAYS built: it drives the observation
        # channel in both reward modes
        alphas = [a.alpha for a in self.agents]
        self.social_calculator = SocialRewardCalculator(
            nb_agents,
            alpha=alphas,
            beta=beta,
            threshold=threshold,
            smoothing=smoothing,
            sigmoid_gain=sigmoid_gain,
            reward_scheme=reward_scheme
        )

        # reward calculator: social (manuscript) or inequity aversion (Hughes 2018)
        if self.reward_mode == "inequity_aversion":
            ia_params = ia_params or {}
            self.reward_calculator = InequityAversionCalculator(nb_agents, **ia_params)
        else:
            self.reward_calculator = self.social_calculator

        self.reset()
        self._alpha_initial = alpha

    @property
    def alpha(self):
        return self._alpha

    def _social_slot_size(self):
        """number of observation slots devoted to the social (emotion) signal."""
        if self.emotion_type == "average":
            return 1
        elif self.emotion_type == "vector":
            return self.nb_agents - 1
        raise ValueError(f"Unknown emotion_type: {self.emotion_type}")

    def _init_agents(self):
        """
        instantiate agents with per-agent see_resource / see_own_meals /
        see_emotions / alpha from agent_configs, falling back to env-level defaults.

        the state size is the SAME for every agent and every condition:
        1 resource slot + 1 own-meals slot + social slot(s) + (optional) 1 time slot.
        conditions differ only in which slots are zeroed out.
        """
        self.agents = []
        self.state_size = 2 + self._social_slot_size() + (1 if self.include_time else 0)

        for idx, config in enumerate(self.agent_configs):
            # read per-agent flags; fall back to env-level defaults
            agent_see_emotions = config.get('see_emotions', self.see_emotions)
            agent_see_resource = config.get('see_resource', self.see_resource)
            agent_see_own_meals = config.get('see_own_meals', self.see_own_meals)
            agent_alpha = config.get('alpha', self._alpha)
            agent_emotion_precision = config.get('emotion_precision', self.emotion_precision)

            # pass all other config keys through, excluding the flags passed
            # as explicit named arguments
            agent_init_kwargs = {
                k: v for k, v in config.items()
                if k not in ('see_emotions', 'see_resource', 'see_own_meals', 'alpha',
                             'emotion_precision')
            }

            agent = self.agent_class(
                state_size=self.state_size,
                action_size=self.n_actions,
                agent_id=idx,
                see_emotions=agent_see_emotions,
                see_resource=agent_see_resource,
                see_own_meals=agent_see_own_meals,
                alpha=agent_alpha,
                emotion_precision=agent_emotion_precision,
                **agent_init_kwargs
            )
            self.agents.append(agent)

    def reset(self):
        self.resource = self.initial_resources
        self.time_step = 0

        # reset per-episode state of the reward calculator (IA smoothed traces)
        self.reward_calculator.reset()

        # NOTE: agent.reset clears meal history and episode state but the
        # replay buffer is deliberately NOT cleared (fix of the old setup,
        # where env.reset wiped agent.memory.buffer every episode)
        for agent in self.agents:
            agent.reset()

        # 03_GT: the first observation is computed AFTER the meal histories are
        # cleared — 02_GT_AIS computed it before, which leaked the previous
        # episode's satiety and emotions into the initial state of every episode
        obs = self.get_observations()
        for idx, agent in enumerate(self.agents):
            agent.current_state = obs[idx]

        return obs

    def get_observations(self):
        """
        return list of observations, one per agent, all of size self.state_size:
          obs = [resource_slot, own_meals_slot] + social_slot(s) + ([time_slot] if include_time)

        gating (per agent):
          - see_resource=False  : resource_slot = 0.0, else r_t / r_0
          - see_own_meals=False : own_meals_slot = 0.0, else the agent's own
            recent consumption rate sum(meal_history)/memory_size — the hidden
            internal state its personal satisfaction and valence derive from
          - social slots: average/vector of the other agents' emotions on the
            scheme's emotion scale (social_calculator.emotion_range), seen
            through perceive_bounded() at the agent's emotion_precision rho
            (rho = 1 exact, rho = 0 uniform; see_emotions=False -> rho = 0).
            the noisy value is re-rounded to round_emotions decimals.

        emotions are always computed for all agents regardless of see_emotions,
        since they feed into the social reward for agents with alpha > 0.
        """
        emotions = self.social_calculator.calculate_emotions(self.agents)
        if self.round_emotions is not None:
            emotions = np.round(emotions, self.round_emotions)

        resource_norm = self.resource / self.initial_resources
        time_norm = self.time_step / self.max_steps if self.include_time else None

        observations = []
        for i, agent in enumerate(self.agents):
            obs = [resource_norm if agent.see_resource else 0.0]

            if agent.see_own_meals:
                obs += [agent.get_recent_meals() / agent.memory_size]
            else:
                obs += [0.0]

            other_emotions = [e for j, e in enumerate(emotions) if j != i]
            if self.emotion_type == "average":
                social = np.array([np.mean(other_emotions)])
            else:
                social = np.array(other_emotions, dtype=float)
            social = self.social_calculator.rescale_emotion(social)
            precision = agent.get_emotion_precision()
            if precision < 1.0:
                low, high = self.social_calculator.emotion_range
                social = perceive_bounded(social, precision, low=low, high=high)
                if self.round_emotions is not None:
                    social = np.round(social, self.round_emotions)
            obs += [float(e) for e in social]

            if self.include_time:
                obs += [time_norm]

            observations.append(np.array(obs, dtype=float))

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

        emotions, personal_reward, empathic_reward, total_reward = \
            self.reward_calculator.calculate_rewards(self.agents,
                                                     immediate_rewards=immediate_rewards,
                                                     actions=actions)

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
            'combined_reward': total_reward,
            'immediate_rewards': immediate_rewards
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
