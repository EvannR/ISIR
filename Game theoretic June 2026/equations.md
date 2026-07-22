## 1. Parameters and Code Mapping

| Parameter | Symbol | Value (defaults) | Code location |
|---|---|---|---|
| Empathy weight | $\alpha$ | varied | `ALPHA` in `main_game_theoretic_new_version.py` |
| Emotion threshold | $\tau$ | 0.5 | `threshold` in `SocialRewardCalculator` |
| Discount factor | $\gamma$ | new param | `gamma` in `SocialRewardCalculator` (replaces `beta`) |
| Memory window | $m$ | **10** (fixed) | `memory_size=10` in `Agent` |
| Initial resources | $E_0$ | 500 | `INITIAL_RESOURCES` |
| Regeneration | $\delta$ | 0.02 | `regen_rate=1.02` in `GameTheoreticEnv` |
| Agents | $N$ | 6 | `NB_AGENTS` |
| Max episode length | $T_{\max}$ | 1000 | `max_steps` in `GameTheoreticEnv` |

**$\beta$ is removed.**  The discount factor $\gamma$ replaces it as the single recency
parameter.  $a = 1/\tau$ and $b = 1/(\tau E_0)$ are fully determined — no free calibration.

---

## 2. Resource Dynamics: Exact Simulation Formula

### 2.1 Resource update

At each step, all agents act simultaneously.  Let $C_t = \sum_{i=1}^N x_i^t$ be the
number of successful consumptions.  The resource updates as:

$$\boxed{E_{t+1} = \max\!\bigl(0,\; \rho\,(E_t - C_t)\bigr)}$$

where $\rho = \texttt{regen\_rate}$ (default $1.02$, so $\delta = \rho - 1 = 0.02$).

### 2.2 Meal success indicator

$$x_i^t = a_i^t \cdot \mathbf{1}[E_t > 0]$$

in **deterministic** mode.  In **stochastic** mode:

$$x_i^t = a_i^t \cdot B_i^t, \qquad B_i^t \sim \mathrm{Bernoulli}\!\left(\frac{E_t}{E_0}\right)$$

The probability of a successful meal decreases linearly as the resource depletes.

### 2.3 Episode termination

$$\text{done}_t = \mathbf{1}[E_t \le 0 \;\vee\; t \ge T_{\max}]$$

The episode ends when the resource pool is fully exhausted **or** the step count
reaches $T_{\max} = 1000$.

---

## 3. Agent Observation (State Space)

After consumption and reward computation, each agent receives an observation built from
the current meal histories.  The state dimension depends on two flags:

| `see_emotions` | `emotion_type` | $s_i^t$ | $\dim(s_i^t)$ |
|---|---|---|---|
| `False` | — | $\mathbf{0}$ | $1$ |
| `True` | `"average"` | $[\hat{e}_i^t]$ | $1$ |
| `True` | `"vector"` | $(e_j^t)_{j \neq i}$ | $N-1$ |

Explicitly:

$$s_i^t =
\begin{cases}
\mathbf{0} & \text{if } \texttt{see\_emotions} = \text{False}\\[4pt]
\left[\hat{e}_i^t\right] & \text{if average mode}\\[4pt]
\left(e_j^t\right)_{j \neq i} & \text{if vector mode}
\end{cases}$$

where $\hat{e}_i^t$ and $e_j^t$ are the empathic signal and individual emotions defined
in §4.2 and §4.3 respectively, computed from meal histories **after** step $t$.

---

## 4. Signal Definitions

### 4.1 Discounted meal-history rate

Let $x_i^{t-k} = \mathbf{1}[\text{agent } i \text{ ate at step } t-k]$, with $k=0$ the most recent step.
The discounted consumption rate over the last $m = 10$ steps is:

$$\boxed{r_i^\gamma = \frac{\displaystyle\sum_{k=0}^{9} \gamma^k\, x_i^{t-k}}{\displaystyle\sum_{k=0}^{9} \gamma^k}}$$

$\gamma \in (0,1]$ is the **meal-history discount** (recency bias within the window).

### 4.2 Emotion signal

$$e_i = \varphi(r_i^\gamma,\,\tau) =
\begin{cases}
\dfrac{r_i^\gamma - \tau}{1-\tau} & r_i^\gamma \geq \tau\\[6pt]
-\dfrac{\tau - r_i^\gamma}{\tau} & r_i^\gamma < \tau
\end{cases}
\quad \in [-1,\,1]$$

$\tau \in (0,1)$ is the **emotion threshold**: the agent feels positive when its recent
consumption rate exceeds $\tau$, and negative otherwise.

### 4.3 Empathic signal

$$\hat{e}_i = \frac{1}{N-1}\sum_{j \neq i} e_j$$

Mean emotion of all other agents, as perceived by agent $i$.

---

## 5. Reward Decomposition and the Tragedy Tension

### 5.1 Combined reward

The empathic penalty is weighted by the agent's own consumption rate when others
feel **worse than the agent itself** — capturing **guilt proportional to relative advantage**.
The scaling transitions smoothly via a sigmoid on $(\hat{e}_i - e_i)$:

$$\boxed{\hat{e}_i^* = \hat{e}_i \cdot \Bigl[r_i^\gamma + (1 - r_i^\gamma)\;\sigma\!\left(\tfrac{\hat{e}_i - e_i}{\lambda}\right)\Bigr]}$$

$$\boxed{R_i = (1-\alpha)\,r_i^\gamma + \alpha\,\hat{e}_i^*}$$

where $\sigma(x) = (1 + e^{-x})^{-1}$ and $\lambda > 0$ controls the sharpness of the transition.

**Limiting behaviour:**

| Regime | $\sigma(\cdot)$ | $\hat{e}_i^*$ | Interpretation |
|---|---|---|---|
| $\hat{e}_i \gg e_i$ | $\to 1$ | $\hat{e}_i$ | Others better off — no guilt scaling |
| $\hat{e}_i = e_i$ | $= 0.5$ | $\hat{e}_i\bigl(\frac{1+r_i^\gamma}{2}\bigr)$ | Equal situation — partial scaling |
| $\hat{e}_i \ll e_i$ | $\to 0$ | $r_i^\gamma \cdot \hat{e}_i$ | Others much worse — full guilt scaling |

**Intuition:** Guilt is absent when others feel the same or better ($\hat{e}_i \geq e_i$, $\sigma \to 1$). It grows continuously as $i$'s relative advantage increases ($\hat{e}_i < e_i$, $\sigma \to 0$), reaching maximum scaling $r_i^\gamma$ when the gap is large. The smooth form avoids the discontinuity that a hard threshold would introduce at $\hat{e}_i = e_i$.

### 5.2 Full substitution

In the limiting guilt regime ($\hat{e}_i \ll e_i$, $\sigma \to 0$), expanding fully:

$$R_i \approx r_i^\gamma\!\left[(1-\alpha) + \frac{\alpha}{N-1}\sum_{j \neq i}\varphi\!\left(r_j^\gamma,\,\tau\right)\right], \qquad r_k^\gamma = \frac{\displaystyle\sum_{l=0}^{9}\gamma^l\,x_k^{t-l}}{\displaystyle\sum_{l=0}^{9}\gamma^l}$$

In the no-guilt limit ($\hat{e}_i \gg e_i$, $\sigma \to 1$), the formula reduces to the unscaled linear combination $(1-\alpha)\,r_i^\gamma + \alpha\,\hat{e}_i$.

### 5.3 The tragedy of the commons tension

At each step, consuming ($a_i = 1$) is locally tempting:

- **Short-term gain:** $x_i^t = 1$ enters the numerator of $r_i^\gamma$, raising personal reward.
- **Collective cost:** $C_t$ increases → $E_{t+1}$ falls → others are less likely to eat in
  future steps → their meal rates $r_j^\gamma$ fall → $e_j$ decreases → $\hat{e}_i - e_i$ grows
  more negative → $\sigma(\hat{e}_i - e_i) \to 0$ → the guilt scaling factor $(1-r_i^\gamma)\sigma$
  shrinks → $\hat{e}_i^*$ approaches $r_i^\gamma \hat{e}_i$, amplifying the penalty
  **proportional to** $r_i^\gamma$: the more $i$ has been eating relative to others, the heavier the guilt.

With $\alpha = 0$ agents maximise $r_i^\gamma$ only: pure selfish incentive → the commons
depletes.  With $\alpha > \alpha_c$ agents learn (via RL) that overconsumption hurts
future empathic reward enough to induce restraint, stabilising $E^*$.

### 5.4 Boundary cases

| Scenario | $e_i$ | $\hat{e}_i$ | Condition | $R_i$ ($\alpha=0$) | $R_i$ ($\alpha=1$) | $R_i$ (general) |
|---|---|---|---|---|---|---|
| All eat | $1$ | $1$ | $\hat{e}_i = e_i$ (unscaled) | $1$ | $1$ | $1$ |
| All starve equally | $-1$ | $-1$ | $\hat{e}_i = e_i$ (unscaled) | $0$ | $-\alpha$ | $(1-\alpha)\cdot 0 + \alpha\cdot(-1) = -\alpha$ |
| I eat, all others starve | $1$ | $-1$ | $\hat{e}_i < e_i$ (guilt) | $1$ | $-1$ | $1-2\alpha$ |

**Note:** When all agents starve equally ($\hat{e}_i = e_i$), the condition $\hat{e}_i \geq e_i$ applies
(no relative disadvantage), so $R_i = (1-\alpha)\,r_i^\gamma + \alpha\,\hat{e}_i \to -\alpha$.
This differs from the previous formulation (where all-starve gave $R_i = 0$): shared suffering
now carries a negative empathic cost rather than being neutralised.
The worst relative case remains eating alone while everyone else starves: $R_i = 1 - 2\alpha$.

---

## 6. Learning Algorithm: Tabular Q-Learning

Each agent independently runs **tabular Q-learning** to maximise its expected
discounted cumulative reward $\sum_t \gamma_\text{RL}^t R_i^t$.

The observation space is small and effectively discrete (emotion values are derived
from binary meal histories over a fixed 10-step window), so function approximation
(e.g. DQN) is unnecessary. A lookup table suffices and converges faster with no
stability concerns.

### 6.1 Notation disambiguation

Two discount parameters appear in the model; they are **distinct**:

| Symbol | Name | Default | Role |
|---|---|---|---|
| $\gamma$ | Meal-history discount | $0.9$ | Shapes reward signal $r_i^\gamma$ |
| $\gamma_\text{RL}$ | RL discount factor | $0.99$ | Shapes value learning (time preference) |

$\alpha$ is the empathy weight; $\alpha_\text{lr}$ denotes the Q-learning rate.

### 6.2 Epsilon-greedy policy

$$\pi_i(s_i^t) =
\begin{cases}
\arg\max_{a \in \{0,1\}} Q(s_i^t, a) & \text{with probability } 1 - \varepsilon_t\\[4pt]
\mathrm{Uniform}\{0,1\} & \text{with probability } \varepsilon_t
\end{cases}$$

Exploration rate decay:

$$\varepsilon_{t+1} = \max\!\bigl(\varepsilon_\text{min},\; \varepsilon_t \cdot \varepsilon_\text{decay}\bigr)$$

### 6.3 Q-learning update

One-step TD target:

$$y_i^t = R_i^t + \gamma_\text{RL}\,\max_{a'} Q(s_i^{t+1}, a')\,(1 - \text{done}_t)$$

Tabular update:

$$Q(s_i^t, a_i^t) \;\leftarrow\; Q(s_i^t, a_i^t) + \alpha_\text{lr}\bigl[y_i^t - Q(s_i^t, a_i^t)\bigr]$$

Unseen states are initialised to $Q(s, a) = 0$.

### 6.4 Hyperparameters

| Hyperparameter | Symbol | Value |
|---|---|---|
| Learning rate | $\alpha_\text{lr}$ | $0.1$ |
| RL discount | $\gamma_\text{RL}$ | $0.99$ |
| Initial exploration | $\varepsilon_0$ | $1.0$ |
| Exploration decay | $\varepsilon_\text{decay}$ | $0.95$ |
| Minimum exploration | $\varepsilon_\text{min}$ | $0.05$ |

---

## 7. Formal Game Tuple $\langle N, S, A, T, R, \Omega, O \rangle$

### $N$ — Agents

$$N = \{1, \dots, 6\}$$

Six agents (`NB_AGENTS = 6`), homogeneous (identical architecture and hyperparameters).

### $S$ — Global state space

The global state at step $t$ is the resource level plus all agents' recent meal histories:

$$S \;=\; \underbrace{[0,\, E_0]}_{\text{resource}} \;\times\; \underbrace{\{0,1\}^{N \times m}}_{\text{meal histories}}$$

Concretely: $s^t = \bigl(E_t,\; (x_i^{t-k})_{i \in N,\, k=0\dots 9}\bigr)$, with $E_0 = 500$ and $m = 10$.

### $A$ — Action space

Per-agent binary choice; joint action $\mathbf{a}^t \in A^N$:

$$A_i = \{0,\,1\}, \qquad A = \{0,1\}^6$$

$a_i = 1$: attempt to eat. $a_i = 0$: abstain.

### $T$ — Transition function

Deterministic mode ($T$ is a Dirac):

$$T(s^t, \mathbf{a}^t) \;\longmapsto\; E_{t+1} = \max\!\bigl(0,\;\rho\,(E_t - C_t)\bigr), \quad C_t = \textstyle\sum_i a_i \cdot \mathbf{1}[E_t > 0]$$

Stochastic mode: $C_t = \sum_i a_i \cdot B_i^t$ with $B_i^t \sim \mathrm{Bernoulli}(E_t / E_0)$, so $T$ is a product of Bernoullis on $S$.

### $R$ — Reward function

$$R_i : S \times A \;\longrightarrow\; \mathbb{R}$$

$$R_i = (1-\alpha)\,r_i^\gamma + \alpha\,\hat{e}_i^*, \qquad \hat{e}_i^* = \hat{e}_i\cdot\!\left[r_i^\gamma + (1-r_i^\gamma)\,\sigma\!\left(\tfrac{\hat{e}_i - e_i}{\lambda}\right)\right]$$

Range: $R_i \in [-(1-\alpha+\alpha\,r_i^\gamma),\; 1]$; bounded in $[-1, 1]$.

### $\Omega$ — Observation space

Depends on the `see_emotions` / `emotion_type` flags:

| `see_emotions` | `emotion_type` | $\Omega_i$ | $\dim$ |
|---|---|---|---|
| `False` | — | $\{0\}$ | $1$ |
| `True` | `"average"` | $[-1,\,1]$ | $1$ |
| `True` | `"vector"` | $[-1,\,1]^{N-1}$ | $5$ |

### $O$ — Observation function

Deterministic: given the global state $s^t$, agent $i$'s observation is computed from meal histories alone (no noise):

$$O(s^t, i) \;=\; s_i^t \;=\;
\begin{cases}
\mathbf{0} & \text{no emotions}\\
\hat{e}_i^t & \text{average mode}\\
(e_j^t)_{j \neq i} & \text{vector mode}
\end{cases}$$

where $e_j^t = \varphi(r_j^\gamma, \tau)$ and $\hat{e}_i^t = \frac{1}{N-1}\sum_{j\neq i} e_j^t$ are fully determined by $(x_j^{t-k})$.
