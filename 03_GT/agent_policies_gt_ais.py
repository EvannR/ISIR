"""
this file defines:
  - Agent                     : base class (meal history, per-agent flags,
                                act() = action played + remembered for step()).
  - perceive_bounded          : noisy Beta observation of a bounded quantity.
  - ReplayBuffer              : experience replay buffer used by DQN agents.
  - DQNNetwork / DQNAgent     : deep Q-network agent with replay + target network.
  - FixedPolicyAgent          : non-learning baseline (random / always exploit / abstain).
  - SocialRewardCalculator    : emotions and reward, journal scheme 'homeostatic_signed',
                                r_i = (1 - alpha_i) * personal_i + alpha_i * empathic_i.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class Agent:
    def __init__(self,
                 agent_id,
                 memory_size=10,
                 see_emotions=True,   # observation dimension: does the agent observe others' emotions?
                 see_resource=True,   # observation dimension: does the agent observe the resource level?
                 see_own_meals=True,  # observation dimension: does the agent observe its own recent consumption rate?
                 alpha=0.5,           # reward dimension: how much does the agent weight others' emotions?
                 emotion_precision=None):  # observation dimension: rho in [0,1]; None -> 1.0 if see_emotions else 0.0
        self.agent_id = agent_id
        self.memory_size = memory_size
        self.meal_history = deque([0.5] * memory_size, maxlen=memory_size) # neutral history (0.5 = satiety threshold): an empty one would make every agent start each episode maximally hungry and distressed
        # self.total_meals = 0
        self.satiety_before_meal = 0.5 # satiety h(t-) in [0,1]: share of meals in the window just BEFORE the last recorded step
        self.see_emotions = see_emotions
        self.see_resource = see_resource
        self.see_own_meals = see_own_meals
        self.alpha = alpha

        if emotion_precision is not None and not 0.0 <= emotion_precision <= 1.0:
            raise ValueError(f"emotion_precision must be in [0, 1], got {emotion_precision}")
        self.emotion_precision = emotion_precision



    def record_meal(self, success: bool):  # , reward: float
        """record whether the agent successfully ate in this timestep."""
        self.satiety_before_meal = sum(self.meal_history) / len(self.meal_history)
        self.meal_history.append(success)
        # if success:
        #     self.total_meals += 1

    def get_recent_meals(self):
        """return the number of recent successful meals in the memory window."""
        return sum(self.meal_history)

    def reset(self, observation=None):
        self.current_state = observation
        self.previous_action = None
        self.meal_history = deque([0.5] * self.memory_size, maxlen=self.memory_size)
        self.satiety_before_meal = 0.5

    # def start_episode(self, state):
    #     """initialize a new episode."""
    #     self.current_state = state
    #     self.previous_action = None

    def act(self, state):
        """
        pick the action actually PLAYED in `state` and remember (state, action):
        the next step() call stores exactly this pair in the transition (state, action) -> (next_state, reward)."""
        action = self.select_action(state)
        self.current_state = state
        self.previous_action = action
        return action

    def end_episode(self):
        """hook called by the runner at the end of each episode """
        pass

    def get_emotion_precision(self):
        """effective precision rho of the social channel (see_emotions is a binary shortcut)."""
        if self.emotion_precision is not None:
            return float(self.emotion_precision)
        return 1.0 if self.see_emotions else 0.0


def precision_to_concentration(precision):
    """
    map the precision rho in [0,1] to the Beta concentration kappa = 2 rho / (1 - rho).
    rho = kappa / (kappa + 2) is the share of the total
    mass (a + b = kappa + 2) carried by the truth: 0 -> uniform, 1 -> Dirac.
    """
    if not 0.0 <= precision <= 1.0:
        raise ValueError(f"precision must be in [0, 1], got {precision}")
    if precision >= 1.0:
        return np.inf
    return 2.0 * precision / (1.0 - precision)


def perceive_bounded(true_value, precision, low=-1.0, high=1.0):  # , rng=None
    """
    noisy observation of a quantity bounded in [low, high]: a Beta draw whose MODE is the
    true value and whose concentration grows with the precision (see precision_to_concentration).
    the mean is pulled towards the middle of the range. precision = 1 returns the true value, without a draw.
    """
    if precision >= 1.0:
        return true_value
    # rng = np.random if rng is None else rng
    kappa = precision_to_concentration(precision)
    u = np.clip((np.asarray(true_value, dtype=float) - low) / (high - low), 0.0, 1.0)
    sample = np.random.beta(1.0 + kappa * u, 1.0 + kappa * (1.0 - u))
    return low + (high - low) * sample


class ReplayBuffer:
    """replay memory of DQN agents: one preallocated array per field, used as a circle."""
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.states = None

    def add(self, state, action, reward, next_state, done):
        """add an experience; once full, it replaces the oldest one."""
        reward = np.atleast_1d(np.asarray(reward, dtype=np.float32))  # one entry per value head (1 = combined reward, 2 = (personal, empathic) for the two-head DQN)
        if self.states is None:  # sizes come from the first experience
            self.states = np.zeros((self.capacity, len(state)), dtype=np.float32)
            self.next_states = np.zeros((self.capacity, len(state)), dtype=np.float32)
            self.actions = np.zeros((self.capacity, 1), dtype=np.int64)
            self.rewards = np.zeros((self.capacity, len(reward)), dtype=np.float32)
            self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """sample a batch of experiences from the buffer."""
        idx = random.sample(range(self.size), min(batch_size, self.size))
        return (torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_states[idx]),
                torch.from_numpy(self.dones[idx]))

    def __len__(self):
        return self.size


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, n_heads=1):
        super(DQNNetwork, self).__init__()
        self.action_size = action_size
        self.n_heads = n_heads

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_heads * action_size)  # shared trunk, one block of action_size outputs per value head

    def forward(self, state):
        """forward pass: state -> hidden layers with relu -> Q-values.
        returns (B, action_size) for one head, (B, n_heads, action_size) otherwise."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)  # raw Q-values, no activation
        if self.n_heads == 1:
            return q
        return q.view(-1, self.n_heads, self.action_size)


class DQNAgent(Agent):
    """
    DQN agent with replay buffer and target network.

    gamma_others: None -> one Q head trained on the combined reward with a single gamma. otherwise the value
    is split along the two reward terms, which the reward is linear in:
        Q(s,a) = (1 - alpha) * Q^p(s,a) + alpha * Q^e(s,a)
    Q^p (personal stream) is discounted with gamma, Q^e (empathic stream) with gamma_others. both heads bootstrap on the SAME action
        a* = argmax_a' Q_target(s', a') of the combined target Q. the action the
    agent would really play. 
    
    with gamma_others < gamma the others weigh alpha now and less and less in the future (see analysis/analysis_01_social_discount_factor.ipynb §4).
    gamma_others == gamma reproduces the single-head targets exactly.
    """
    def __init__(self, state_size, action_size, agent_id=0, hidden_size=64,
                 learning_rate=0.001, gamma=0.99, gamma_others=None, epsilon=1.0,
                 epsilon_decay=0.99, epsilon_min=0.01, batch_size=64,
                 update_target_every=10, buffer_capacity=100000,
                 see_emotions=True, see_resource=True, see_own_meals=True,
                 alpha=0.5, emotion_precision=None, learn_every=4):
        super().__init__(agent_id, see_emotions=see_emotions,
                         see_resource=see_resource, see_own_meals=see_own_meals,
                         alpha=alpha, emotion_precision=emotion_precision)
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gamma_others = gamma_others
        self.two_heads = gamma_others is not None
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0
        self.learn_every = learn_every # one learning update every learn_every stored transitions (1 = every step)
        self.transitions = 0

        n_heads = 2 if self.two_heads else 1
        self.head_weights = torch.tensor([1.0 - alpha, alpha]).view(1, 2, 1) # head weights (1 - alpha, alpha) and per-head discounts, shaped to broadcast over (B, n_heads, action_size)
        self.head_gammas = torch.tensor([gamma, gamma_others if self.two_heads else gamma]).view(1, 2)

        self.policy_network = DQNNetwork(state_size, action_size, hidden_size, n_heads=n_heads)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size, n_heads=n_heads)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=buffer_capacity) # buffer large enough to hold many episodes: it persists across episodes

        self.current_state = None
        self.previous_action = None

    def combined_q(self, q):
        """combined action values: identity for one head, (1-alpha) Q^p + alpha Q^e for two."""
        if not self.two_heads:
            return q
        return (self.head_weights * q).sum(dim=1)

    def select_action(self, state):
        """select an action using epsilon-greedy exploration."""
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad(): # no eval()/train() toggle needed: DQNNetwork has no Dropout/BatchNorm, so train/eval steps have zero effect on the forward pass here
            action_values = self.combined_q(self.policy_network(state_tensor))

        return int(np.argmax(action_values.cpu().data.numpy()))

    def learn(self, experiences):
        """learn from a batch of experiences using DQN loss."""
        states, actions, rewards, next_states, dones = experiences

        if not self.two_heads:
            q_expected = self.policy_network(states).gather(1, actions)                          # Q(s,a) from the policy network for the actions actually taken
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)    # max_a' Q_target(s',a') from the frozen target network
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))                    # TD target: r + gamma * max Q_target(s',a') * (1 - done)
        else:
            # rewards: (B, 2) = (personal, empathic); dones: (B, 1)
            idx = actions.unsqueeze(1).expand(-1, 2, 1)                         # (B, 2, 1)
            q_expected = self.policy_network(states).gather(2, idx).squeeze(2)  # (B, 2)
            with torch.no_grad():
                q_next = self.target_network(next_states)                       # (B, 2, A)
                
                a_star = self.combined_q(q_next).argmax(dim=1, keepdim=True)    # one shared bootstrap action, chosen on the COMBINED target Q
                q_next_a = q_next.gather(2, a_star.unsqueeze(1).expand(-1, 2, 1)).squeeze(2)
                q_targets = rewards + self.head_gammas * q_next_a * (1 - dones) # per-head TD target: r_h + gamma_h * Q_h,target(s', a*) * (1 - done)

        
        loss = F.mse_loss(q_expected, q_targets) # mean over batch AND heads: for two heads, half the sum of the two head MSEs

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        self.steps += 1 # periodically sync target network weights with policy network
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def step(self, next_state, reward, done, reward_parts=None):
        """
        store the transition (state, action) recorded by act() -> next_state and
        learn once the buffer is large enough.
        reward is the combined reward (single head); the two-head agent needs
        reward_parts = (personal, empathic) instead.
        """
        if self.two_heads:
            if reward_parts is None:
                raise ValueError("two-head DQN (gamma_others set) needs reward_parts=(personal, empathic)")
            stored_reward = [float(reward_parts[0]), float(reward_parts[1])]
        else:
            stored_reward = float(reward)

        if self.current_state is not None and self.previous_action is not None:
            self.memory.add(self.current_state, self.previous_action, stored_reward, next_state, done)
            self.transitions += 1

            # only start learning once enough experiences are collected
            if len(self.memory) > self.batch_size and self.transitions % self.learn_every == 0:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def end_episode(self):
        """decay exploration rate once per episode (was per step in the old setup)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class FixedPolicyAgent(Agent):
    """
    non-learning baseline agent, interchangeable with DQNAgent in the environment and the runner.

    policies:
      - 'random'         : uniform random action each step
      - 'always_exploit' : action 1 every step (greedy upper bound on depletion speed)
      - 'always_abstain' : action 0 every step (degenerate sustainability reference)
    """
    def __init__(self, state_size, action_size, agent_id=0, policy='random',
                 memory_size=10, see_emotions=False, see_resource=False,
                 see_own_meals=False, alpha=0.0, emotion_precision=None, **kwargs):
        super().__init__(agent_id, memory_size=memory_size,
                         see_emotions=see_emotions, see_resource=see_resource,
                         see_own_meals=see_own_meals, alpha=alpha,
                         emotion_precision=emotion_precision)
        if policy not in ('random', 'always_exploit', 'always_abstain'):
            raise ValueError(f"Unknown fixed policy: {policy}")
        self.state_size = state_size
        self.action_size = action_size
        self.policy = policy
        self.epsilon = 0.0  # exposed for logging symmetry with learning agents
        self.current_state = None
        self.previous_action = None

    def select_action(self, state):
        if self.policy == 'random':
            return int(np.random.choice(self.action_size))
        elif self.policy == 'always_exploit':
            return 1
        return 0

    def step(self, next_state, reward, done, reward_parts=None):
        """no learning."""
        pass


class SocialRewardCalculator:
    """
    emotions and reward of the agents:
        r_i = (1 - alpha_i) * personal_i + alpha_i * empathic_i

    reward_scheme selects the two terms:
      - 'homeostatic_signed' (journal model): personal_i = meal_i * (1 - h_i), a meal is
        worth less the more sated the agent is before it; empathic_i = mean emotion of
        the others on [-1, 1], so their distress is a penalty. r_i in [-alpha_i, 1].
      - 'homeostatic': same personal term, empathic_i on [0, 1]. r_i in [0, 1].
      - 'legacy_ais2026': conference formula, personal_i = beta * last meal
        + (1 - beta) * history, empathic_i on [-1, 1]. r_i in [-alpha_i, 1].

    calculate_emotions() returns the raw emotions in [-1, 1]; rescale_emotion() puts them
    on the scheme's scale (EMOTION_RANGES), shared by the empathic term and the observation.
    alpha is one float for all agents, or one value per agent.
    """
    # 'homeostatic_signed': homeostatic personal term, empathic term on [-1, 1]
    # as in the AIS paper, so that the others' distress is a penalty
    REWARD_SCHEMES = ('homeostatic', 'homeostatic_signed', 'legacy_ais2026')
    # emotion scale of each scheme: shared by the empathic term and the observation
    EMOTION_RANGES = {'homeostatic': (0.0, 1.0),
                      'homeostatic_signed': (-1.0, 1.0),
                      'legacy_ais2026': (-1.0, 1.0)}

    def __init__(self, nb_agents, alpha=0.5, beta=0.5, threshold=0.5,
                 smoothing='linear', sigmoid_gain=5.0, reward_scheme='homeostatic_signed'):
        if reward_scheme not in self.REWARD_SCHEMES:
            raise ValueError(f"Unknown reward_scheme: {reward_scheme}")
        if smoothing != 'linear':  # the sigmoid is commented out below
            raise ValueError(f"Unknown smoothing: {smoothing}")
        self.reward_scheme = reward_scheme
        self.emotion_range = self.EMOTION_RANGES[reward_scheme]
        self.nb_agents = nb_agents
        # normalise alpha to always be a per-agent list internally
        if isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == nb_agents, "alpha list length must match nb_agents"
            self.alpha = [float(a) for a in alpha]
        else:
            self.alpha = [float(alpha)] * nb_agents
        self.beta = beta
        self.threshold = threshold
        self.smoothing = smoothing
        self.sigmoid_gain = sigmoid_gain

    def _consumption_rate(self, agent):
        """return average consumption rate [0,1] based on binary meal_history."""
        return sum(agent.meal_history) / len(agent.meal_history)

    def _linear_emotion(self, rate):
        """linear mapping from rate to emotion signal [-1,1] based on threshold."""
        t = self.threshold
        if rate >= t:
            return (rate - t) / (1 - t) if t < 1 else 1.0
        return - (t - rate) / t if t > 0 else -1.0

    # def _sigmoid_emotion(self, rate):
    #     """sigmoid mapping from rate to emotion signal [-1,1] centered at threshold."""
    #     g = self.sigmoid_gain
    #     t = self.threshold
    #     norm = (rate - t) / (1 - t) if rate >= t else (rate - t) / t
    #     exp_val = np.exp(-g * norm)
    #     return 2.0 / (1.0 + exp_val) - 1.0

    def emotion_from_rate(self, rate):
        """compute emotion signal from consumption rate using the chosen smoothing."""
        # if self.smoothing == 'sigmoid':
        #     return self._sigmoid_emotion(rate)
        return self._linear_emotion(rate)

    def calculate_personal_satisfaction(self, agent):
        """compute personal satisfaction from last meal and history."""
        last = 1.0 if agent.meal_history[-1] > 0 else 0.0
        hist = sum(agent.meal_history) / len(agent.meal_history)
        return self.beta * last + (1 - self.beta) * hist

    def calculate_homeostatic_reward(self, agent):
        """meal value scaled by the remaining need: meal * (1 - satiety before the meal)."""
        last = 1.0 if agent.meal_history[-1] > 0 else 0.0
        return last * (1.0 - agent.satiety_before_meal)

    def rescale_emotion(self, emotion):
        """map a raw emotion in [-1, 1] onto the scheme's emotion scale."""
        low, high = self.emotion_range
        return low + (high - low) * (np.asarray(emotion, dtype=float) + 1.0) / 2.0

    def calculate_emotions(self, agents):
        """return list of emotion signals for each agent."""
        return [self.emotion_from_rate(self._consumption_rate(a)) for a in agents]

    def calculate_rewards(self, agents):
        """
        return four lists, one value per agent:
        - emotions : emotion signals, on [-1, 1]
        - personal : personal terms, on [0, 1]
        - empathic : mean emotion of the others, on the scheme's scale
        - total    : (1 - alpha_i) * personal_i + alpha_i * empathic_i
        """
        homeostatic = self.reward_scheme in ('homeostatic', 'homeostatic_signed')
        if homeostatic:
            personal = [self.calculate_homeostatic_reward(a) for a in agents]
        else:
            personal = [self.calculate_personal_satisfaction(a) for a in agents]

        emotions = self.calculate_emotions(agents)

        """ empathic reward: mean emotion of all other agents, on the scheme's emotion scale (the same scale as the emotion observation) """
        empathic_reward = []
        for idx, emo in enumerate(emotions):
            others_emo = np.mean([e for j, e in enumerate(emotions) if j != idx])
            empathic_reward.append(float(self.rescale_emotion(others_emo)))

        """ total reward: per-agent weighted combination """
        total = [
            (1 - self.alpha[i]) * personal[i] + self.alpha[i] * empathic_reward[i]
            for i in range(self.nb_agents)
        ]

        return emotions, personal, empathic_reward, total
