# `01_GT_` — Tragédie des communs multi-agents (setup game-theoretic)

Ce dossier contient une variante **game-theoretic** de la simulation d'empathie /
tragédie des communs. `N` agents partagent un pool de ressources fini et décident,
à chaque pas de temps et de façon indépendante, d'**exploiter** (action = 1) ou de
**s'abstenir** (action = 0). La ressource diminue avec l'exploitation et peut se
régénérer.

## Ce qui est spécifique à ce dossier

Par rapport aux autres versions de la simulation, `01_GT_` sépare **trois axes de
manipulation expérimentale strictement orthogonaux** — chacun contrôlé par un
paramètre distinct, sans interaction dans le code :

| Axe | Paramètre | Effet |
|-----|-----------|-------|
| **Voir les émotions des autres** | `see_emotions` (bool) | l'agent observe l'état émotionnel des autres (moyenne ou vecteur), ou rien |
| **Voir la ressource** | `see_resources` (bool) | l'agent observe le **niveau de ressource partagé** (spécificité récente de ce dossier) |
| **Prise en compte des autres** | `alpha` (0–1) | poids de la composante empathique dans le reward |

Point clé : **`see_resources` n'entre QUE dans l'observation** — jamais dans le
calcul du reward, jamais dans `SocialRewardCalculator`, jamais dans `see_emotions`
ni `alpha`. C'est un axe perceptif purement additionnel.

### Structure de l'observation

L'observation de chaque agent est construite dans
[`GameTheoreticEnv.get_observations`](env_game_theoretic.py) :

```
obs = [ ...partie émotion... ,  (resource_signal) ]
```

- **Partie émotion** (inchangée) :
  - `see_emotions=False` → placeholder `[0.0]`
  - `see_emotions=True, emotion_type="average"` → `[moyenne des émotions des autres]`
  - `see_emotions=True, emotion_type="vector"` → `[émotions des autres]`
- **Signal ressource** (ajouté seulement si `see_resources=True`) :
  `resource / initial_resources ∈ [0, 1]`, arrondi comme les émotions
  (`round_emotions`). C'est un **scalaire global** (identique pour tous les agents),
  ce qui garde un espace d'états fini pour le Q-learning tabulaire.

**Non-régression garantie** : avec `see_resources=False`, le `state_size` et les
observations sont strictement identiques au comportement historique.

### Reward

Le reward combine satisfaction personnelle et signal empathique, calculé par
[`SocialRewardCalculator`](agent_policies_game_theoretic.py) :

```
combined = (1 - alpha) * personal_satisfaction + alpha * empathic_reward
```

où `empathic_reward` = moyenne des émotions des **autres** agents. Les émotions
sont dérivées du taux de consommation récent (fenêtre `meal_history`) via un
mapping `linear` ou `sigmoid` centré sur un `threshold`.

## Fichiers

| Fichier | Rôle |
|---------|------|
| [`env_game_theoretic.py`](env_game_theoretic.py) | environnement `GameTheoreticEnv` : état, observations, dynamique de la ressource, calcul des rewards |
| [`agent_policies_game_theoretic.py`](agent_policies_game_theoretic.py) | agents (`QAgent` tabulaire, `DQNAgent`), `ReplayBuffer`, et `SocialRewardCalculator` |
| [`main_game_theoretic_new_version.py`](main_game_theoretic_new_version.py) | point d'entrée : orchestration des runs multi-conditions, seeding, écriture CSV incrémentale |
| [`jerome_analysis_first.ipynb`](jerome_analysis_first.ipynb) | notebook d'analyse des résultats |
| `value_cal.py` | script de brouillon (vérification manuelle de la formule de reward) — non utilisé par la simulation |

## Lancer une simulation

```bash
python main_game_theoretic_new_version.py
```

Les CSV sont écrits sous `../GT_simulation_2_empathy/`. Le nom de fichier encode la
condition, dont **`SEE_EMOTIONS` et `SEE_RESOURCES`** pour tracer l'axe perceptif :

```
results_<sim>_<episodes>_<agent>_<emotion_type>_<see_emotions>_<see_resources>_<alpha>_<beta>_...
```

### Constantes de configuration principales

Définies en tête de [`main_game_theoretic_new_version.py`](main_game_theoretic_new_version.py) :

| Constante | Description |
|-----------|-------------|
| `NB_AGENTS` | nombre d'agents |
| `EPISODE_NUMBER`, `MAX_STEPS` | nombre d'épisodes et de pas par épisode |
| `INITIAL_RESOURCES` | taille initiale du pool partagé |
| `ENVIRONMENT_TYPE` | `"deterministic"` ou `"stochastic"` (succès d'exploitation ∝ ressource restante) |
| `AGENT_TO_TEST` | `"DQN"` ou `"QLearning"` |
| `SEE_EMOTIONS` | axe 1 — voir les émotions des autres |
| `SEE_RESOURCES` | axe 2 — voir le niveau de ressource |
| `ALPHA` | axe 3 — poids empathique dans le reward |
| `EMOTION_TYPE` | `"average"` ou `"vector"` |
| `BETA`, `SMOOTHING`, `THRESHOLD`, `SIGMOID_GAIN` | paramètres du mapping émotionnel |
| `EMOTION_DECIMALS` | arrondi des signaux émotion **et** ressource |

## Notes / limitations connues

- **Mode `emotion_type="vector"`** : incohérence préexistante entre le `state_size`
  posé dans `_init_agents` (`nb_agents`) et la longueur réelle des observations
  (`nb_agents − 1`). Les runs utilisent `"average"`, donc non exercé — à corriger si
  l'on passe au mode vecteur.
- **Log step-level** : l'en-tête CSV suppose `NB_AGENTS` colonnes d'observation ;
  avec `see_resources=True`, chaque cellule contient un vecteur de longueur 2. Non
  bloquant car les runs tournent avec `save_steps=False`.
