# 03_GT — version journal (dérivée de 02_GT_AIS)

Changements par rapport à 02_GT_AIS (vérifiés dans
[analysis/analysis_00_reward_observation_checks.ipynb](analysis/analysis_00_reward_observation_checks.ipynb)) :

| # | Changement | Où |
|---|-----------|-----|
| 1 | **Récompense homéostatique** (`REWARD_SCHEME = "homeostatic"`) : p_i = repas_i·(1 − h_i), avec h_i la satiété (part de repas sur les 10 derniers pas) **avant** le pas courant : seuls les repas rapportent (s'abstenir = 0, repas affamé = 1, repas repu = 0). Terme empathique ẽ_i = (ē + 1)/2, émotion moyenne des autres sur [0, 1]. Donc r_i = (1−α)·p_i + α·ẽ_i ∈ [0, 1] quel que soit α (l'ancienne formule donnait [−α, 1]), jamais négatif. L'ancienne formule reste disponible : `"legacy_ais2026"`. | `SocialRewardCalculator`, `Agent.record_meal` |
| 2 | **Observation graduée des émotions** : `emotion_precision` ρ ∈ [0, 1] par agent. Le signal social observé est exprimé sur la **même échelle [0, 1] que le terme empathique ẽ** (c'est la même grandeur) et tiré d'une Beta de mode égal à la vraie valeur et de concentration κ = 2ρ/(1−ρ) : ρ = 0 → uniforme, ρ = 1 → valeur exacte. `see_emotions=True/False` équivaut à ρ = 1/0 (**False donne désormais un bruit uniforme, et non plus 0**). | `perceive_bounded`, `env.get_observations` |
| 3 | Bug corrigé : `env.reset()` calculait la première observation avant de vider les historiques de repas (fuite de l'épisode précédent au pas 0). | `env.reset` |
| 4 | **Discount factor social** `gamma_others` (γ_o) : DQN à deux têtes, Q = (1−α)·Q^p[γ] + α·Q^e[γ_o], les deux têtes bootstrappant sur la même action (choisie sur le Q combiné). Têtes combinées **brutes**, sans normalisation. `gamma_others=None` → modèle à une tête inchangé ; γ_o = γ reproduit exactement les cibles à une tête. Réservé au DQN avec `reward_mode="social"`. Nouvelle expérience `exp7_social_discount` : γ_o ∈ {0.99, 0.95, 0.9, 0.8, 0, 0.995, 0.999} à α = 0.5, plus des contrôles « α apparié » (γ unique, même part totale d'empathie). Justification : [analysis_01](analysis/analysis_01_social_discount_factor.ipynb). | `DQNAgent`, `make_condition`, `EXPERIMENTS["exp7_social_discount"]` |
| 5 | Bug corrigé : l'action stockée dans le replay buffer venait d'un second tirage ε-greedy dans `agent.step()`, différent de l'action jouée (≈47 % de transitions fausses à ε = 1), et la première transition de chaque épisode était perdue. Les agents exposent maintenant `act()` (joue et mémorise l'action), et `step()` ne fait plus que stocker et apprendre. **Les runs 03_GT ne sont donc pas comparables bit à bit avec 02_GT_AIS.** | `Agent.act`, `run_step` |

| 6 | **Observation des émotions individuelles** (`EMOTION_TYPE = "vector"`, défaut 03_GT depuis le 02/10/2026) : le signal social n'est plus la moyenne des autres (1 case) mais l'émotion de **chacun des 5 autres agents** (5 cases, ordre fixe = indice des agents sans soi-même), sur [0, 1]. Le bruit ρ est tiré indépendamment pour chaque case. L'observation passe de 3 à 7 dimensions. Le terme empathique de la **reward reste la moyenne** des autres. Attention : à ρ égal, un agent qui moyenne ses 5 cases bruitées retrouve la moyenne avec moins de bruit qu'en mode `"average"`, donc les ρ ne sont pas comparables entre les deux modes. | `EMOTION_TYPE` (main), `env.get_observations` |
| 7 | **Évaluations précoces et poids sauvegardés** : une évaluation greedy avant tout entraînement (`episode = -1`) et après chacun des 20 premiers épisodes (`--eval-dense-first`), avec l'état du générateur aléatoire restauré pour ne pas modifier l'entraînement ; poids initiaux et finaux des DQN dans `<run>_weights.pt`. | `run_simulation`, `run_eval_episode`, `save_weights` |
| 8 | **Reward de retenue** (`REWARD_SCHEME = "homeostatic_restraint"`, défaut depuis le 05/10/2026 ; le premier batch exp1 est en `"homeostatic"`) : terme personnel inchangé, le terme empathique devient un **bonus d'abstention** ẽ_i = abstention_i · max(0, s_i − s̄_{-i}), satisfactions sur [0, 1] prises **avant** le pas (ce que l'agent observait). S'abstenir vaut mieux que manger ssi α·(s_i − s̄_{-i}) > (1−α)·(1 − h_i) : α = 0 mange toujours ; pour α ≤ 0,5 un agent affamé (h < 0,5) mange quoi qu'il arrive, au-delà de α = 0,5 même un agent affamé peut être poussé à s'abstenir. r reste dans [0, 1]. Une tentative de repas ratée ne reçoit pas le bonus (les actions sont transmises au calculateur). | `SocialRewardCalculator.restraint_bonus`, `env.step` |

Les CSV de résumé ont trois colonnes de plus : `emotion_precision`, `gamma_others` (vide = une seule tête) et `reward_scheme`.

---

# 02_GT_AIS — Expériences de révision AIS 2026 (soumission 1885)

Code de re-run pour répondre aux reviews AIS 2026 du manuscrit
*"Model-free multi-agent reinforcement learning simulation of the influence of
empathy on the sustainability of agents' consumption in a common-pool resource
dilemma"*.

Dérivé de `empathy_tragedy/game_theoretic_matrix_setup/` — **le code original
n'est pas modifié**, tout se passe dans ce dossier. Le suivi de l'avancement des
expériences est dans [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md).

---

## Correspondance reviews → changements

| # | Critique (reviewer) | Changement dans ce dossier |
|---|---------------------|----------------------------|
| 1 | R1 : « extreme sensory bottleneck » — les agents ne voient jamais r_t, le problème devient un bandit sans état | Observation vectorielle `[r_t/r_0, conso récente propre, signal social, (temps)]` avec gating par agent (`see_resource`, `see_own_meals`, `see_emotions`). Le slot « conso propre » = sum(meal_history)/10, l'état interne dont dérivent la satisfaction personnelle et la valence émise (règle le problème de récompense non-markovienne). Dimension d'entrée **identique dans toutes les conditions** (l'information est mise à zéro, pas l'architecture). → `env_gt_ais.get_observations()` |
| 2 | R1 : « acute algorithmic simplifications » | (a) Le replay buffer **persiste entre épisodes** (l'ancien `env.reset()` le vidait à chaque épisode). (b) Epsilon décroît **par épisode** (`agent.end_episode()`), pas par pas de temps — avant, l'exploration s'effondrait dès le 1er épisode sur 500. |
| 3 | R1 + R2 : « missing essential experimental baselines » | Agents à politique fixe sans apprentissage : `random`, `always_exploit` (+ `always_abstain` disponible). → `FixedPolicyAgent` |
| 4 | R2 : Hughes et al. 2018 (inequity aversion) trouve le résultat inverse sur l'égalité | Baseline `InequityAversionCalculator` : récompense subjective u_i = r_i − (α/(N−1))·envie − (β/(N−1))·culpabilité sur traces lissées, mêmes observations que la condition D. |
| 5 | R2 : ANOVA sur 4 seeds seulement (F(1,12)) | `NUM_RUNS_PER_CONDITION = 20`. Seeds déterministes : `BASE_SEED + condition_idx*1000 + sim`. |
| 6 | R2 : « does the fairness reversal persist with … different γ? » | `exp5_gamma_robustness` : conditions A et D à γ ∈ {0.9, 0.999}. |
| 7 | (analyse inégalité) | Les CSV de résumé contiennent désormais les repas par agent et par épisode (`meals_i`) et le coefficient de Gini (`gini_meals`), directement exploitables pour les stats de fairness. |

Ce qui ne change **pas** : la formule de récompense sociale
r_i = (1−α_i)·personnel_i + α_i·empathique_i, le calcul des émotions
(meal history → seuil → lissage linéaire), l'environnement stochastique,
6 agents, 500 ressources initiales, 500 épisodes, 1000 pas max.

---

## Les cinq expériences

| Expérience | Conditions | Répond à |
|------------|-----------|----------|
| `exp1_resource_obs` | 2×2 (see_emotions × alpha), tous avec `see_resource=True` | R1 — le re-run principal demandé |
| `exp2_replication_no_res` | 2×2 du papier (`see_resource=False`, `see_own_meals=False`), entraînement corrigé, 20 seeds | Pont entre anciens et nouveaux résultats : les effets du papier survivent-ils aux correctifs d'entraînement ? |
| `exp3_fixed_baselines` | `random_policy`, `always_exploit` | Points de référence non-apprenants |
| `exp4_inequity_aversion` | DQN + récompense IA (Hughes 2018), observations comme D | Contraste littérature demandé par R2 |
| `exp5_gamma_robustness` | A et D à γ = 0.9 et 0.999 | Robustesse du renversement de fairness |
| `exp6_graded_alpha` | Study 2 du papier : α ∈ {0, 0.25, 0.5, 0.75, 0.99}, observation active, avec r_t | Re-run dose-réponse (TH3/TH4) sous régime corrigé |

**Question scientifique clé de exp1 :** une fois que les agents voient r_t,
l'effet « observation des émotions » (C vs A) survit-il ? Le signal émotionnel
était jusqu'ici la seule information disponible ; c'est le cœur de la réponse à
Hughes et al.

---

## Fichiers

```
02_GT_AIS/
  agent_policies_gt_ais.py  # agents (DQN, Q-tab, politiques fixes) + calculateurs de récompense
  env_gt_ais.py             # environnement avec observation [ressource, social, (temps)]
  main_gt_ais.py            # runner par expériences nommées (EXPERIMENTS / RUN_EXPERIMENTS)
  README.md                 # ce fichier
  EXPERIMENT_TRACKER.md     # suivi : fait dans le papier / à refaire / jamais fait
  NOTES_ANALYSE_REVIEWS.md  # analyse croisée reviews × papier × code (discussion 24/07/2026)
  results/<experiment>/     # CSV de sortie + sidecars JSON de config
```

## Lancer

```bash
cd 02_GT_AIS
python main_gt_ais.py --smoke        # sanity check : 2 épisodes, 1 seed, toutes conditions
python main_gt_ais.py                # défaut : 1 réplication, exp1 + exp6 (les études du papier)
python main_gt_ais.py --runs 7       # 7 réplications par condition
python main_gt_ais.py --exp exp2_replication_no_res exp3_fixed_baselines
python main_gt_ais.py --eval-every 0 # désactiver les épisodes d'évaluation greedy
```

### Répartition sur les 3 machines (objectif : 20 seeds/condition)

Chaque indice de simulation correspond à un seed déterministe unique → aucune
collision possible entre machines :

```bash
# machine 1 : simulations 0 à 6
python main_gt_ais.py --runs 7 --sim-start 0
# machine 2 : simulations 7 à 13
python main_gt_ais.py --runs 7 --sim-start 7
# machine 3 : simulations 14 à 19
python main_gt_ais.py --runs 6 --sim-start 14
```

À la fin, rapatrier les trois dossiers `csv_results/` dans un seul (les noms de
fichiers encodent condition + seed, aucun conflit).

### Évaluation greedy

Tous les `--eval-every` épisodes (défaut 10), un épisode supplémentaire est
joué avec ε = 0 et **sans apprentissage**, loggé avec `phase='eval'` dans le
CSV. **Ce sont ces lignes qu'il faut utiliser pour les analyses finales** — les
lignes `phase='train'` contiennent le bruit d'exploration.

### Divers

- Reprise : les runs dont le CSV de résumé existe déjà sont sautés — on peut
  relancer la même commande après interruption sans dupliquer.
- Chaque run écrit `<label>_seed<seed>_episode_summary.csv` + un sidecar
  `<label>_seed<seed>_config.json` contenant la paramétrisation complète.
- Ordre de grandeur : ~1h40 par run DQN de 500 épisodes (+ ~10 % pour l'éval),
  soit ~15 h par machine pour le lot prioritaire exp1+exp6 en 7 seeds.
