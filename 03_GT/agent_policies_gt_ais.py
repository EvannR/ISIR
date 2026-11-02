"""
agent policies for the AIS-2026 revision experiments (02_GT_AIS).

derived from empathy_tragedy/game_theoretic_matrix_setup/agent_policies_game_theoretic.py
with the changes requested by the AIS 2026 reviews:

  1. epsilon decays PER EPISODE (end_episode()), not per step.  with the old
     per-step decay (0.995 per step, min 0.01), exploration collapsed within the
     first episode out of 500.
  2. the replay buffer is NOT cleared between episodes (see env_gt_ais.reset).
  3. new fixed-policy baseline agents (random / always_exploit / always_abstain)
     that share the Agent interface but never learn — reference points requested
     by both reviewers ("missing essential experimental baselines").
  4. new InequityAversionCalculator implementing the subjective reward of
     Hughes et al. 2018 ("Inequity aversion improves cooperation in
     intertemporal social dilemmas", NeurIPS) as a literature baseline.
  5. agents carry a see_resource flag (observation of r_t / r_0), in addition to
     see_emotions (social signal) and alpha (reward weighting) — answer to R1's
     "extreme sensory bottleneck".

03_GT (journal version) changes on top of the AIS-2026 code:

  6. HOMEOSTATIC REWARD (reward_scheme='homeostatic', new default).
     the value of a meal decreases with the agent's satiety h_i, the fraction
     of meals in its memory window BEFORE the current step:
         personal_i = meal_i(t) * (1 - h_i(t-))                 in [0, 1]
         empathic_i = (mean_{j != i} e_j + 1) / 2               in [0, 1]
         r_i        = (1 - alpha_i) * personal_i + alpha_i * empathic_i
     only meals are rewarded: abstaining scores 0, a meal when starving 1, a
     meal when fully sated 0. both terms live on [0, 1], so r_i is bounded by
     [0, 1] for EVERY alpha (the AIS formula ranged over [-alpha, 1]); rewards
     are never negative, so ending the episode early is never rewarded.
     the emotion OBSERVATION uses the same [0, 1] scale (see 7), so the
     empathic term of the reward and the observed signal are one quantity.
     the AIS formula is kept as reward_scheme='legacy_ais2026'.
     variant 'homeostatic_restraint' (03_GT default since 05/10/2026): the
     empathic term is replaced by a restraint bonus, abstain_i *
     max(0, s_i - s̄_{-i}), so abstaining can beat eating when the agent is
     better off than the others (see SocialRewardCalculator).
  7. GRADED EMOTION OBSERVATION. the binary see_emotions flag is replaced by a
     per-agent emotion_precision rho in [0, 1]: the observed social signal is
     drawn from a Beta distribution on the emotion scale of the reward scheme
     ([0, 1] homeostatic, [-1, 1] legacy) whose MODE is the true value and
     whose concentration is kappa = 2 rho / (1 - rho):
         rho = 0 -> Beta(1, 1): uniform on the scale, independent of the truth
         rho = 1 -> Dirac at the true value (exact observation, no RNG draw)
     see_emotions=True/False remains a shortcut for rho = 1 / rho = 0.
  8. SOCIAL DISCOUNT FACTOR (DQNAgent gamma_others). the personal and empathic
     reward streams can be discounted with different factors through a
     two-head DQN, Q = (1 - alpha) Q^p[gamma] + alpha Q^e[gamma_others]; see
     DQNAgent and analysis/analysis_01_social_discount_factor.ipynb.
  9. STORED ACTION = PLAYED ACTION. the runner used to draw the played action
     with select_action() while step() drew a second epsilon-greedy action and
     stored THAT one in the transition (~47 % mislabelled transitions at
     epsilon = 1), and the first transition of every episode was dropped.
     agents now expose act(), which plays the action and remembers it, and
     step() only stores/learns. still present in 02_GT_AIS and the AIS runs.

this module defines:
  - Agent                     : base class (meal history, per-agent flags,
                                act() = action played + remembered for step()).
  - perceive_bounded          : noisy Beta observation of a bounded quantity.
  - ReplayBuffer              : experience replay buffer used by DQN agents.
  - QAgent                    : tabular Q-learning agent.
  - DQNNetwork / DQNAgent     : deep Q-network agent with replay + target network.
  - FixedPolicyAgent          : non-learning baseline (random / always exploit / abstain).
  - SocialRewardCalculator    : emotion-based reward, homeostatic or legacy scheme,
                                r_i = (1 - alpha_i) * personal_i + alpha_i * empathic_i.
  - InequityAversionCalculator: Hughes et al. 2018 subjective reward baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


class Agent:
    def __init__(self,
                 agent_id,
                 memory_size=10,
                 see_emotions=True,   # observation dimension: does this agent observe others' emotions?
                 see_resource=True,   # observation dimension: does this agent observe the resource level?
                 see_own_meals=True,  # observation dimension: does this agent observe its own recent consumption rate?
                 alpha=0.5,           # reward dimension: how much does this agent weight others' emotions?
                 emotion_precision=None):  # observation dimension: rho in [0,1]; None -> 1.0 if see_emotions else 0.0
        self.agent_id = agent_id
        self.memory_size = memory_size
        # neutral history (0.5 = satiety threshold): an empty one would make every
        # agent start each episode maximally hungry and distressed (decided 14/10/2026)
        self.meal_history = deque([0.5] * memory_size, maxlen=memory_size)
        self.total_meals = 0
        # satiety h(t-) in [0,1]: share of meals in the window just BEFORE the
        # last recorded step — the homeostatic state the meal's value depends on
        self.satiety_before_meal = 0.5
        self.see_emotions = see_emotions
        self.see_resource = see_resource
        self.see_own_meals = see_own_meals
        self.alpha = alpha
        if emotion_precision is not None and not 0.0 <= emotion_precision <= 1.0:
            raise ValueError(f"emotion_precision must be in [0, 1], got {emotion_precision}")
        self.emotion_precision = emotion_precision

    def get_emotion_precision(self):
        """effective precision rho of the social channel (see_emotions is the binary shortcut)."""
        if self.emotion_precision is not None:
            return float(self.emotion_precision)
        return 1.0 if self.see_emotions else 0.0

    def record_meal(self, success: bool, reward: float):
        """record whether the agent successfully ate in this timestep."""
        self.satiety_before_meal = sum(self.meal_history) / len(self.meal_history)
        self.meal_history.append(success)
        if success:
            self.total_meals += 1

    def get_recent_meals(self):
        """return the number of recent successful meals in the memory window."""
        return sum(self.meal_history)

    def reset(self, observation=None):
        self.current_state = observation
        self.previous_action = None
        self.meal_history = deque([0.5] * self.memory_size, maxlen=self.memory_size)
        self.satiety_before_meal = 0.5

    def start_episode(self, state):
        """initialize a new episode."""
        self.current_state = state
        self.previous_action = None

    def act(self, state):
        """
        pick the action actually PLAYED in `state` and remember (state, action):
        the next step() call stores exactly this pair in the transition. the
        runner must use act() during training; select_action() stays a pure
        policy query (used by the greedy evaluation episodes).
        """
        action = self.select_action(state)
        self.current_state = state
        self.previous_action = action
        return action

    def end_episode(self):
        """hook called by the runner at the end of each episode (no-op by default)."""
        pass


def precision_to_concentration(precision):
    """
    map the precision rho in [0,1] to the Beta concentration kappa = 2 rho / (1 - rho).

    kappa counts pseudo-observations of the true value added to a uniform
    Beta(1, 1) prior, so rho = kappa / (kappa + 2) is the share of the total
    mass (a + b = kappa + 2) carried by the truth: 0 -> uniform, 1 -> Dirac.
    """
    if not 0.0 <= precision <= 1.0:
        raise ValueError(f"precision must be in [0, 1], got {precision}")
    if precision >= 1.0:
        return np.inf
    return 2.0 * precision / (1.0 - precision)


def perceive_bounded(true_value, precision, low=-1.0, high=1.0, rng=None):
    """
    noisy observation of a quantity bounded in [low, high].

    the true value x is rescaled to u in [0, 1] and the observation is drawn
    from Beta(1 + kappa*u, 1 + kappa*(1-u)), whose mode is exactly u for every
    kappa, then rescaled back to [low, high]. the support never leaves
    [low, high], so the observation channel has the same range at every
    precision. precision = 1 returns the true value without drawing from the
    RNG, so exact-observation runs consume the same random stream as before.
    true_value may be a scalar or an array (one independent draw per entry).
    """
    if precision >= 1.0:
        return true_value
    rng = np.random if rng is None else rng
    kappa = precision_to_concentration(precision)
    u = np.clip((np.asarray(true_value, dtype=float) - low) / (high - low), 0.0, 1.0)
    sample = rng.beta(1.0 + kappa * u, 1.0 + kappa * (1.0 - u))
    return low + (high - low) * sample


Experience =namedtuple('Experience', ['state',
                                       'action',
                                       'reward',
                                       'next_state',
                                       'done'])


class ReplayBuffer:
    """replay memory buffer for DQN experience replay."""
    def __init__(self, capacity=10000):
        # plain list + circular write pointer instead of deque: random.sample()
        # needs O(1) indexing, which deque cannot offer (block-linked-list layout)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """add an experience to the buffer, normalising types for torch."""
        if isinstance(action, (list, tuple)):
            action = np.array([action[0]], dtype=np.int64)
        elif isinstance(action, np.ndarray):
            action = np.array([action.item()], dtype=np.int64)
        else:
            action = np.array([int(action)], dtype=np.int64)

        state = np.array(state, dtype=np.float32)
        # reward is stored as a 1-D vector: one entry per value head
        # (1 = combined reward, 2 = (personal, empathic) for the two-head DQN)
        reward = np.atleast_1d(np.asarray(reward, dtype=np.float32))
        next_state = np.array(next_state, dtype=np.float32)
        if not isinstance(done, bool):
            done = bool(done.item()) if isinstance(done, np.ndarray) else bool(done)

        experience = Experience(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()  # (B, n_heads)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.tensor([[int(e.done)] for e in experiences], dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QAgent(Agent):
    def __init__(self, state_size, action_size, agent_id=0, learning_rate=0.1,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.99,
                 epsilon_min=0.01, memory_size=10,
                 see_emotions=True, see_resource=True, see_own_meals=True,
                 alpha=0.5, emotion_precision=None):
        super().__init__(agent_id, memory_size=memory_size,
                         see_emotions=see_emotions, see_resource=see_resource,
                         see_own_meals=see_own_meals, alpha=alpha,
                         emotion_precision=emotion_precision)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}

        self.current_state = None
        self.previous_action = None

    def get_state_key(self, state):
        """convert the state into a hashable key for Q-table lookup."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, (list, tuple)):
            return tuple(state)
        else:
            return (state,)

    def get_q_values(self, state):
        """return q-values for the given state, initialising to zeros if unseen."""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def select_action(self, state):
        """select an action using epsilon-greedy exploration."""
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_size))
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done):
        """update the Q-table with one-step TD target."""
        state_key = self.get_state_key(state)

        q_values = self.get_q_values(state)
        # if terminal state, future value is zero
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_size)
        # TD target = immediate reward + discounted best future Q
        target = reward + self.gamma * np.max(next_q_values)
        q_values[action] = q_values[action] + self.learning_rate * (target - q_values[action])
        self.q_table[state_key] = q_values

    def step(self, next_state, reward, done, reward_parts=None):
        """learn from the transition (state, action) recorded by act() -> next_state.
        reward_parts is accepted for interface symmetry with DQNAgent and ignored."""
        if self.current_state is not None and self.previous_action is not None:
            self.learn(self.current_state, self.previous_action, reward, next_state, done)

    def end_episode(self):
        """decay exploration rate once per episode (was per step in the old setup)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, n_heads=1):
        super(DQNNetwork, self).__init__()
        self.action_size = action_size
        self.n_heads = n_heads

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # shared trunk, one block of action_size outputs per value head
        self.fc3 = nn.Linear(hidden_size, n_heads * action_size)

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

    gamma_others (03_GT, social discount factor): None -> one Q head trained on
    the combined reward with a single gamma (the AIS model). otherwise the value
    is split along the two reward terms, which the reward is linear in:
        Q(s,a) = (1 - alpha) * Q^p(s,a) + alpha * Q^e(s,a)
    Q^p (personal stream) is discounted with gamma, Q^e (empathic stream, the
    others' emotion) with gamma_others. both heads bootstrap on the SAME action
    a* = argmax_a' Q_target(s', a') of the combined target Q — the action the
    agent would really play; per-head maxima would describe a purely selfish and
    a purely altruistic agent whose sum matches no actual policy. heads are
    combined RAW (not rescaled by 1 - gamma): with gamma_others < gamma the
    others weigh alpha now and less and less in the future (decision of
    02/10/2026, see analysis/analysis_01_social_discount_factor.ipynb §4).
    gamma_others == gamma reproduces the single-head targets exactly.
    """
    def __init__(self, state_size, action_size, agent_id=0, hidden_size=64,
                 learning_rate=0.001, gamma=0.99, gamma_others=None, epsilon=1.0,
                 epsilon_decay=0.99, epsilon_min=0.01, batch_size=64,
                 update_target_every=10, buffer_capacity=100000,
                 see_emotions=True, see_resource=True, see_own_meals=True,
                 alpha=0.5, emotion_precision=None, learn_every=1):
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
        # one learning update every learn_every stored transitions (1 = every step)
        self.learn_every = learn_every
        self.transitions = 0

        n_heads = 2 if self.two_heads else 1
        # head weights (1 - alpha, alpha) and per-head discounts, shaped to
        # broadcast over (B, n_heads, action_size)
        self.head_weights = torch.tensor([1.0 - alpha, alpha]).view(1, 2, 1)
        self.head_gammas = torch.tensor([gamma, gamma_others if self.two_heads else gamma]).view(1, 2)

        self.policy_network = DQNNetwork(state_size, action_size, hidden_size, n_heads=n_heads)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size, n_heads=n_heads)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        # buffer large enough to hold many episodes: it persists across episodes
        self.memory = ReplayBuffer(capacity=buffer_capacity)

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

        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        # no eval()/train() toggle needed: DQNNetwork has no Dropout/BatchNorm,
        # so train/eval mode has zero effect on the forward pass here
        with torch.no_grad():
            action_values = self.combined_q(self.policy_network(state_tensor))

        return int(np.argmax(action_values.cpu().data.numpy()))

    def learn(self, experiences):
        """learn from a batch of experiences using DQN loss."""
        states, actions, rewards, next_states, dones = experiences

        if not self.two_heads:
            # Q(s,a) from the policy network for the actions actually taken
            q_expected = self.policy_network(states).gather(1, actions)
            # max_a' Q_target(s',a') from the frozen target network
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            # TD target: r + gamma * max Q_target(s',a') * (1 - done)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        else:
            # rewards: (B, 2) = (personal, empathic); dones: (B, 1)
            idx = actions.unsqueeze(1).expand(-1, 2, 1)                      # (B, 2, 1)
            q_expected = self.policy_network(states).gather(2, idx).squeeze(2)  # (B, 2)
            with torch.no_grad():
                q_next = self.target_network(next_states)                      # (B, 2, A)
                # one shared bootstrap action, chosen on the COMBINED target Q
                a_star = self.combined_q(q_next).argmax(dim=1, keepdim=True)   # (B, 1)
                q_next_a = q_next.gather(2, a_star.unsqueeze(1).expand(-1, 2, 1)).squeeze(2)
                # per-head TD target: r_h + gamma_h * Q_h,target(s', a*) * (1 - done)
                q_targets = rewards + self.head_gammas * q_next_a * (1 - dones)

        # mean over batch AND heads: for two heads, half the sum of the two head MSEs
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodically sync target network weights with policy network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """store an experience in the replay buffer after type normalisation."""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        if isinstance(done, (list, tuple, np.ndarray)):
            done = bool(done[0])
        else:
            done = bool(done)

        self.memory.add(state, action, reward, next_state, done)

    def step(self, next_state, reward, done, reward_parts=None):
        """
        store the transition (state, action) recorded by act() -> next_state and
        learn once the buffer is large enough.
        reward is the combined reward (single head); the two-head agent needs
        reward_parts = (personal, empathic) instead.
        """
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)

        if self.two_heads:
            if reward_parts is None:
                raise ValueError("two-head DQN (gamma_others set) needs reward_parts=(personal, empathic)")
            stored_reward = [float(reward_parts[0]), float(reward_parts[1])]
        else:
            stored_reward = float(reward)

        if self.current_state is not None and self.previous_action is not None:
            self.store_transition(self.current_state, self.previous_action, stored_reward, next_state, done)
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
    non-learning baseline agent. shares the Agent interface (select_action /
    step / start_episode / end_episode) so it is interchangeable with QAgent
    and DQNAgent in the environment and the runner.

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
    calculate rewards based on agents' consumption and configurable emotional smoothing.

    combines a personal term with an empathic signal (mean emotion of all
    other agents) into a per-agent reward:
        r_i = (1 - alpha_i) * personal_i + alpha_i * empathic_i.

    reward_scheme selects the two terms:
      - 'homeostatic' (03_GT default): personal_i = meal_i * (1 - h_i(t-)), only
        meals are rewarded and a meal is worth less the more sated the agent
        already is; empathic_i = mean others' emotion on the [0, 1] scale.
        both in [0, 1] -> r_i in [0, 1] for all alpha.
      - 'homeostatic_restraint': same personal term; the empathic term becomes
        a RESTRAINT BONUS, earned only by choosing to abstain while being
        better off than the others:
            empathic_i = abstain_i * max(0, s_i(t-) - s̄_{-i}(t-))
        with s the emotion on the [0, 1] scale, taken BEFORE the step (what
        the agent observed when it decided). abstaining beats eating iff
            alpha * (s_i - s̄_{-i}) > (1 - alpha) * (1 - h_i)
        so alpha = 0 always eats, and for alpha <= 0.5 a hungry agent
        (h_i < 0.5) eats whatever the others feel; above alpha = 0.5 even a
        hungry agent can be pushed to abstain. r_i stays in [0, 1].
      - 'legacy_ais2026': the manuscript formula, personal_i = beta * last meal
        + (1 - beta) * history in [0, 1], empathic_i = mean others' emotion in
        [-1, 1] -> r_i in [-alpha_i, 1].
    calculate_emotions() always returns the raw emotions in [-1, 1];
    rescale_emotion() maps them onto the scheme's emotion scale
    (EMOTION_RANGE), used both by the empathic term and by the observation.

    alpha can be a single float (same for all agents) or a list of per-agent values.
    """
    # 'homeostatic_signed': homeostatic personal term, empathic term on [-1, 1]
    # as in the AIS paper, so that the others' distress is a penalty
    REWARD_SCHEMES = ('homeostatic', 'homeostatic_restraint', 'homeostatic_signed',
                      'legacy_ais2026')
    # emotion scale of each scheme: shared by the empathic term and the observation
    EMOTION_RANGES = {'homeostatic': (0.0, 1.0), 'homeostatic_restraint': (0.0, 1.0),
                      'homeostatic_signed': (-1.0, 1.0),
                      'legacy_ais2026': (-1.0, 1.0)}

    def __init__(self, nb_agents, alpha=0.5, beta=0.5, threshold=0.7,
                 smoothing='linear', sigmoid_gain=10.0, reward_scheme='homeostatic'):
        if reward_scheme not in self.REWARD_SCHEMES:
            raise ValueError(f"Unknown reward_scheme: {reward_scheme}")
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

    def reset(self):
        """no per-episode state (kept for interface symmetry with the IA calculator)."""
        pass

    def _consumption_rate(self, agent):
        """return average consumption rate [0,1] based on binary meal_history."""
        if not agent.meal_history:
            return 0.0
        return sum(agent.meal_history) / len(agent.meal_history)

    def _linear_emotion(self, rate):
        """linear mapping from rate to emotion signal [-1,1] based on threshold."""
        t = self.threshold
        if rate >= t:
            return (rate - t) / (1 - t) if t < 1 else 1.0
        return - (t - rate) / t if t > 0 else -1.0

    def _sigmoid_emotion(self, rate):
        """sigmoid mapping from rate to emotion signal [-1,1] centered at threshold."""
        g = self.sigmoid_gain
        t = self.threshold
        norm = (rate - t) / (1 - t) if rate >= t else (rate - t) / t
        exp_val = np.exp(-g * norm)
        return 2.0 / (1.0 + exp_val) - 1.0

    def emotion_from_rate(self, rate):
        """compute emotion signal from consumption rate using the chosen smoothing."""
        if self.smoothing == 'sigmoid':
            return self._sigmoid_emotion(rate)
        return self._linear_emotion(rate)

    def calculate_personal_satisfaction(self, agent):
        """compute personal satisfaction from last meal and history."""
        if agent.meal_history:
            last = 1.0 if agent.meal_history[-1] > 0 else 0.0
            hist = sum(agent.meal_history) / len(agent.meal_history)
        else:
            last, hist = 0.0, 0.0
        return self.beta * last + (1 - self.beta) * hist

    def calculate_homeostatic_reward(self, agent):
        """meal value scaled by the remaining need: meal * (1 - satiety before the meal)."""
        last = 1.0 if agent.meal_history and agent.meal_history[-1] > 0 else 0.0
        return last * (1.0 - agent.satiety_before_meal)

    def rescale_emotion(self, emotion):
        """map a raw emotion in [-1, 1] onto the scheme's emotion scale."""
        low, high = self.emotion_range
        return low + (high - low) * (np.asarray(emotion, dtype=float) + 1.0) / 2.0

    def calculate_emotions(self, agents):
        """return list of emotion signals for each agent."""
        return [self.emotion_from_rate(self._consumption_rate(a)) for a in agents]

    def restraint_bonus(self, agents, actions):
        """empathic term of 'homeostatic_restraint': abstain_i * max(0, s_i - s̄_{-i}),
        emotions on the [0, 1] scale computed from the satiety BEFORE this step."""
        pre = [float(self.rescale_emotion(self.emotion_from_rate(a.satiety_before_meal)))
               for a in agents]
        bonus = []
        for i, act in enumerate(actions):
            others = np.mean([s for j, s in enumerate(pre) if j != i])
            bonus.append(max(0.0, pre[i] - others) if act == 0 else 0.0)
        return bonus

    def calculate_rewards(self, agents, immediate_rewards=None, actions=None):
        """
        compute and return:
        - emotions   : list of emotion signals ([-1,1])
        - personal   : list of personal terms ([0,1])
        - empathic   : list of empathic terms: mean emotion of the others on the
                       scheme's emotion scale ([0,1] homeostatic, [-1,1] legacy),
                       or the restraint bonus ('homeostatic_restraint', [0,1])
        - total      : list of total rewards, using per-agent alpha:
                       total_i = (1 - alpha_i) * personal_i + alpha_i * empathic_i

        immediate_rewards is accepted (and ignored) for interface symmetry with
        InequityAversionCalculator. actions (0 = abstain, 1 = exploit) is
        required by 'homeostatic_restraint' only.
        """
        homeostatic = self.reward_scheme in ('homeostatic', 'homeostatic_restraint',
                                             'homeostatic_signed')
        if homeostatic:
            personal = [self.calculate_homeostatic_reward(a) for a in agents]
        else:
            personal = [self.calculate_personal_satisfaction(a) for a in agents]

        emotions = self.calculate_emotions(agents)

        if self.reward_scheme == 'homeostatic_restraint':
            if actions is None:
                raise ValueError("'homeostatic_restraint' needs the agents' actions")
            empathic_reward = self.restraint_bonus(agents, actions)
        else:
            # empathic reward: mean emotion of all other agents, on the scheme's
            # emotion scale (the same scale as the emotion observation)
            empathic_reward = []
            for idx, emo in enumerate(emotions):
                others_emo = np.mean([e for j, e in enumerate(emotions) if j != idx])
                empathic_reward.append(float(self.rescale_emotion(others_emo)))

        # total reward: per-agent weighted combination
        total = [
            (1 - self.alpha[i]) * personal[i] + self.alpha[i] * empathic_reward[i]
            for i in range(self.nb_agents)
        ]

        return emotions, personal, empathic_reward, total


class InequityAversionCalculator:
    """
    inequity-aversion subjective reward of Hughes et al. 2018 (NeurIPS),
    "Inequity aversion improves cooperation in intertemporal social dilemmas".

    each agent i keeps a temporally smoothed reward trace
        e_i(t) = lambda_ia * e_i(t-1) + r_i(t)
    and receives the subjective reward
        u_i(t) = r_i(t)
                 - (alpha_ia / (N-1)) * sum_j max(e_j - e_i, 0)   # disadvantageous inequity ("envy")
                 - (beta_ia  / (N-1)) * sum_j max(e_i - e_j, 0)   # advantageous inequity ("guilt")

    in Hughes et al., advantageous-inequity aversion (beta_ia) is the term that
    promotes cooperation, hence the default alpha_ia=0.0, beta_ia=0.5.
    the trace is reset at every episode start (env.reset -> calculator.reset()).
    """
    def __init__(self, nb_agents, alpha_ia=0.0, beta_ia=0.5, lambda_ia=0.95):
        self.nb_agents = nb_agents
        self.alpha_ia = alpha_ia
        self.beta_ia = beta_ia
        self.lambda_ia = lambda_ia
        self.smoothed = np.zeros(nb_agents, dtype=float)

    def reset(self):
        """reset the smoothed reward traces at episode start."""
        self.smoothed[:] = 0.0

    def calculate_rewards(self, agents, immediate_rewards=None, actions=None):
        """
        actions is accepted (and ignored) for interface symmetry with
        SocialRewardCalculator.

        compute the subjective inequity-averse reward from the immediate
        (environment) rewards of this step.

        returns the same 4-tuple structure as SocialRewardCalculator so the
        environment can use either calculator interchangeably:
        - emotions : [0.0] * N (no emotion machinery in this baseline)
        - personal : the raw immediate rewards r_i(t)
        - empathic : the (negative) inequity penalty applied to each agent
        - total    : the subjective rewards u_i(t)
        """
        if immediate_rewards is None:
            raise ValueError("InequityAversionCalculator requires immediate_rewards")

        r = np.array(immediate_rewards, dtype=float)
        self.smoothed = self.lambda_ia * self.smoothed + r

        n = self.nb_agents
        penalties = []
        for i in range(n):
            envy = sum(max(self.smoothed[j] - self.smoothed[i], 0.0)
                       for j in range(n) if j != i)
            guilt = sum(max(self.smoothed[i] - self.smoothed[j], 0.0)
                        for j in range(n) if j != i)
            penalty = (self.alpha_ia / (n - 1)) * envy + (self.beta_ia / (n - 1)) * guilt
            penalties.append(-penalty)

        total = [float(r[i] + penalties[i]) for i in range(n)]
        emotions = [0.0] * n

        return emotions, r.tolist(), penalties, total
