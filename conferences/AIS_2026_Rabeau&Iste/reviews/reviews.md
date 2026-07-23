# AIS 2026 — Reviews

**Submission 1885**
**Title:** Model-free multi-agent reinforcement learning simulation of the influence of empathy on the sustainability of agents' consumption in a common-pool resource dilemma

---

## Review 1

**SCORE:** 2 (accept)

**TEXT:**

The conceptual synthesis of psychological empathy into MARL reward terms is refreshing, and the sub-additive/paradoxical results are intellectually stimulating. However, the manuscript suffers from acute algorithmic simplifications, missing essential experimental baselines, major presentation inconsistencies between text and figures, and an inadequate state/action representation that borders on trivializing the RL element. The paper can become a strong contribution if these technical issues are comprehensively addressed.

The environment is formalized as a partially observable stochastic game where the true state space S includes the current resource level r_t ≥ 0. However, Section 2.5 explicitly notes that "all agents are trained via Deep Q-Network (DQN), each taking the scalar observation o_i as sole input."

If the sole input to the network is o_i (the mean scalar affective signal of the peers), the agents have zero information regarding the true state of the resource (r_t). This reduces the underlying setup from a true Reinforcement Learning problem to a stateless, historical multi-agent contextual bandit scenario or a single-state POMDP.

The authors must clarify how an agent can learn policy shifts tied directly to resource degradation over time if the actual resource level or a local proxy of the resource level is entirely omitted from their observation tensor. If r_t or a time step indicator is truly omitted, the networks are merely mapping an average external mood to a binary option. The authors must re-run the experiments providing agents with an observation vector that contains both the resource status (or local availability) and the social signal, or explicitly address this extreme sensory bottleneck.

**Critical Misalignments Between Figures and Text.** There is a massive, highly confusing contradiction between the textual description of the experimental results and the uploaded figures.

**Very Important:** good text structure, but undermined by crucial typographical mistakes and legend confusion. Please fix these mistakes.

Revision is required before acceptance can be considered.

---

## Review 2

**SCORE:** -1 (weak reject)

**TEXT:**

The authors train six DQN agents in a common-pool resource game, factorially manipulating whether agents perceive peers' affective signals (cognitive empathy) and whether those signals enter the reward (motivational empathy), finding that empathy extends resource lifetime but increases inequality.

Approximately, the Research Question is: "Does empathy promote sustainable and fair use of shared resources?" This is operationalized as: "Do DQN agents whose reward includes a weighted mean of peers' broadcast valence signals deplete a simulated common pool more slowly, and distribute access more equally, than selfish agents?"

The 2x2 factorial dissociation between observation and reward is clear and complemented by the graded weights used for empathy. The treatment of specifically what kind of empathy we want to define in the experiment is also helpful. Result about the cognitive and motivational empathy causing higher inequality is surprising and compelling. Limitations of simulation like binary decision model are clearly acknowledge and the future direction is interesting.

The four hypotheses put together with the simulation have a falsifiability problem. Confirmations of TH1 and TH3 seem to support the empathy altruism theory but when TH2/TH4 are contradicted, §4.1 (p.13) reinterprets the empathy-altruism hypothesis as targeting welfare rather than equality. It is difficult for this simulation to validate empathy theory per se. Authors could reframe to a claim of sufficiency, i.e. this formalization of empathy is sufficient to produce sustainability in this environment, with increased inequality as an unanticipated consequence.

Moreover, there are some prior threads of literature that could be engaged more.
("To our knowledge, no existing work has yet proposed a way to model empathetic processes within a MARL framework" (§1.2, p.3))
- Hughes et al. 2018, "Inequity aversion improves cooperation in intertemporal social dilemmas" (NeurIPS) which has the opposite finding of your result. To consider in rewrite: How does your mechanism differ from Hughes, and would you expect their equality result to survive under your affective-signal channel?
- Prosocial reward shaping in social dilemmas is a kind of empathy and worth contrasting with what you do
- Emotion-mediated cooperation in RL specifically also exists, see https://arxiv.org/abs/1906.10918

ANOVAs only done on four replications per condition (Table 2; F(1,12)) and should be done with more seeds (cheap fix) given the surprise in the claim. To consider in rewrite: Does the fairness reversal persist with more seeds and different γ?

Result raises interesting question on the difference between sustainability and equity and under what circumstances they are at odds. Reframing and replications needed go beyond a camera-ready revision. Encouragement to resubmit. Medium confidence - I may be unaware of additional related work in this area.

---

## Summary / Action Items

| # | Reviewer | Score | Core ask |
|---|----------|-------|----------|
| 1 | R1 | 2 (accept) | Fix observation bottleneck (add resource level/proxy to state), fix figure/text mismatches, fix typos/legend confusion |
| 2 | R2 | -1 (weak reject) | Reframe falsifiability of hypotheses (sufficiency claim vs. theory validation), engage Hughes et al. 2018 and related literature, increase seeds/replications for ANOVA (currently 4 per condition) |




for rew-submission : https://easychair.org/conferences/overview?a=37322171
