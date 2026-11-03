"""
this file defines:
  - GameTheoreticEnv : the shared resource of the commons dilemma. at each step, it applies the
                       actions, computes meals and rewards, then builds each agent's observation.

observation of an agent, same size in all conditions (the network architecture never changes):
    [ resource, own_meals, social slot(s), (time) ]
  - resource  : r_t / r_0, 0.0 if see_resource=False.
  - own_meals : sum(meal_history) / memory_size, 0.0 if see_own_meals=False.
  - social    : emotions of the others ('vector': one slot each, 'average': their mean), on the
                reward scheme's scale ([-1, 1] for 'homeostatic_signed'), seen through
                perceive_bounded at the precision rho: rho = 1 exact, rho = 0 uniform noise
                (see_emotions=False means rho = 0, not 0.0).
  - time      : t / max_steps, only if include_time=True.
"""

import numpy as np
from agent_policies_gt_ais import (DQNAgent, FixedPolicyAgent,
                                   SocialRewardCalculator,
                                   perceive_bounded)


class GameTheoreticEnv:
    """
    multi-agent environment where each agent can exploit a shared resource or not.
    each agent has its own see_resource, see_own_meals, see_emotions, emotion_precision and alpha
    (set via agent_configs), the arguments of the same name are only the defaults.
    """
    def __init__(self, nb_agents,
                 initial_resources=500,
                 regen_rate=1.0,
                 env_type="stochastic",
                 emotion_type="vector",
                 see_emotions=True,
                 see_resource=True,
                 see_own_meals=True,
                 alpha=0.5,
                 emotion_precision=None,
                 beta=0.5,
                 agent_class=DQNAgent,
                 agent_configs=None,
                 threshold=0.5,
                 smoothing='linear',
                 sigmoid_gain=5.0,
                 round_emotions=2,
                 include_time=False,
                 max_steps=1000,
                 reward_scheme="homeostatic_signed"):

        self.nb_agents = nb_agents
        self.initial_resources = initial_resources
        self.regen_rate = regen_rate
        self.env_type = env_type              # "deterministic" or "stochastic"
        self.emotion_type = emotion_type      # "average" or "vector"
        self.see_emotions = see_emotions
        self.see_resource = see_resource
        self.see_own_meals = see_own_meals
        self._alpha = alpha
        self.emotion_precision = emotion_precision
        self.n_actions = 2                    # 0: abstain, 1: exploit
        self.round_emotions = round_emotions
        self.include_time = include_time
        self.max_steps = max_steps

        self.agent_class = agent_class
        self.agent_configs = agent_configs or [{} for _ in range(nb_agents)]

        self._init_agents() # first, the calculator reads the per-agent alphas

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

        self.reset()

    def _social_slot_size(self):
        """number of observation slots of the social signal."""
        if self.emotion_type == "average":
            return 1
        elif self.emotion_type == "vector":
            return self.nb_agents - 1
        raise ValueError(f"Unknown emotion_type: {self.emotion_type}")

    def _init_agents(self):
        """create the agents from agent_configs, with the arguments of the env as defaults."""
        self.agents = []
        self.state_size = 2 + self._social_slot_size() + (1 if self.include_time else 0)

        for idx, config in enumerate(self.agent_configs):
            agent_see_emotions = config.get('see_emotions', self.see_emotions)
            agent_see_resource = config.get('see_resource', self.see_resource)
            agent_see_own_meals = config.get('see_own_meals', self.see_own_meals)
            agent_alpha = config.get('alpha', self._alpha)
            agent_emotion_precision = config.get('emotion_precision', self.emotion_precision)

            agent_init_kwargs = {
                k: v for k, v in config.items()
                if k not in ('see_emotions', 'see_resource', 'see_own_meals', 'alpha',
                             'emotion_precision')
            } # the other keys go straight to the agent class

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
        """start a new episode: the replay buffers are kept, they persist across episodes."""
        self.resource = self.initial_resources
        self.time_step = 0

        for agent in self.agents:
            agent.reset()

        obs = self.get_observations() # after agent.reset: otherwise the satiety and emotions of the previous episode leak into the first state
        for idx, agent in enumerate(self.agents):
            agent.current_state = obs[idx]

        return obs

    def get_observations(self):
        """return one observation per agent, see the top of this file. the noisy social slots are rounded again."""
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
        apply one 0/1 action per agent. the agents eat at the same time, from the same stock.
        returns: next_observations, rewards, done, info.
        """
        consumed = 0
        immediate_rewards = []

        for idx, act in enumerate(actions):
            reward = 0.0
            success = False
            if act == 1 and self.resource > 0:
                if self.env_type == "stochastic":
                    prob = self.resource / self.initial_resources # success gets less likely as the resource runs out
                    success = np.random.rand() < prob
                else:
                    success = True
                if success:
                    reward = 1.0
                    consumed += 1

            self.agents[idx].record_meal(success)
            immediate_rewards.append(reward)

        emotions, personal_reward, empathic_reward, total_reward = \
            self.social_calculator.calculate_rewards(self.agents)

        self.resource = max(0.0, (self.resource - consumed) * self.regen_rate)
        self.time_step += 1
        next_obs = self.get_observations()
        done = self.resource <= 0 # the episode ends when the resource is depleted

        info = {
            'emotions': emotions,
            'personal_reward': personal_reward,
            'empathic_reward': empathic_reward,
            'combined_reward': total_reward,
            'immediate_rewards': immediate_rewards # 0/1 meal of each agent
        }

        return next_obs, total_reward, done, info

    # def get_agent_meal_stats(self, agent_idx):
    #     """return recent and total meals for one agent."""
    #     a = self.agents[agent_idx]
    #     return {
    #         'recent_meals': a.get_recent_meals(),
    #         'total_meals': a.total_meals,
    #         'meal_history': list(a.meal_history)
    #     }

    # def get_all_agents_meal_stats(self):
    #     """return meal stats for all agents."""
    #     return [self.get_agent_meal_stats(i) for i in range(self.nb_agents)]
