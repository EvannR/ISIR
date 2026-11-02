# AGY Tracker — Propositions d'Expériences & Pistes de Recherche (03_GT)

Ce document propose des extensions et nouvelles expériences pour la **version journal (03_GT)**, formulées à partir des dynamiques observées dans la simulation, des verrous méthodologiques actuels (collapsus de la reward homéostatique pure, retenue codée vs émergente, renversement d'équité / *fairness reversal*), et de la littérature scientifique en MARL, neurosciences cognitives, théorie des jeux et gestion des biens communs (CPR).

---

## 1. Diagnostic de la Simulation 03_GT & Tensions Théoriques

1. **Le paradoxe du gâteau fixe (Absence de régénération)** : Dans `env_gt_ais.py`, `regen_rate = 1.0` sans terme d'accroissement naturel ($R_{t+1} = \max(0, R_t - \sum a_i)$). Le nombre total de repas possibles est rigoureusement borné à $R_0 = 500$. La soutenabilité ne correspond donc qu'à un ralentissement de l'épuisement inéluctable, et non à un équilibre de récolte durable (Maximum Sustainable Yield, Ostrom 1990).
2. **L'échec de la reward homéostatique pure (`homeostatic`)** : Comme observé lors du premier batch exp1, manger rapporte $p_i = 1 - h_i \ge 0$ et s'abstenir rapporte $0$. Manger domine strictement à chaque pas immédiat, et la somme temporelle des émotions des autres est constante sur l'épisode ($\approx 83{,}2$). L'empathie non différenciée n'incite pas à la retenue.
3. **Le dilemme de la retenue codée (`homeostatic_restraint`)** : Le bonus $\tilde{e}_i = \text{abstain}_i \cdot \max(0, s_i - \bar{s}_{-i})$ injecte directement la retenue comme un axiome de récompense au lieu de la faire émerger des interactions futures (via $\gamma$).
4. **Le mystère du renversement d'équité (*Fairness Reversal*)** : Pourquoi l'empathie ($\alpha > 0$) augmente-t-elle l'inégalité (Gini) ? Est-ce un effet d'exploitation altruiste (quelques agents se sacrifient pendant que d'autres s'accaparent la ressource) ?
5. **Désalignement émotionnel** : L'émotion émise reste monotone croissante avec le taux de consommation ($f \to 1$), alors que la reward personnelle de l'agent est maximale à l'optimum homéostatique ($f = 0{,}5$).

---

## 2. Pistes Expérimentales Proposées

### Axe A — Dynamique Environnementale & Émergence Naturelle de la Retenue
*Objectif : Faire émerger la retenue comme un équilibre de Nash / politique optimale sans avoir à la coder en dur dans la fonction de reward.*

#### Exp A.1 — Régénération Renouvelable (CPR Logistique)
- **Concept** : Introduire une dynamique classique de ressource renouvelable : $R_{t+1} = \min(R_{\max}, R_t + r \cdot R_t (1 - R_t / K) - \sum C_i)$.
- **Hypothèse** : Si la ressource se régénère, la retenue permet d'atteindre un état stationnaire infini (rendement maximal soutenable, MSY). S'abstenir n'est plus un pur sacrifice, mais un investissement intertemporel qui maximise la somme cumulée des repas pour soi et pour autrui.
- **Paramètres** : $r \in \{0.05, 0.1, 0.2\}$, capacité de charge $K = 500$, seuil critique de collapse.
- **Métriques** : Taux d'extinction de la ressource, nombre total de pas jusqu'à épuisement ou convergence à l'état stable ($T_{\max} = 1000$).
- **Ancrage Littérature** : Ostrom (1990), Perolat et al. (2017, DeepMind CPR), Leibo et al. (2017).

#### Exp A.2 — Goulot d'Étranglement & Concurrence par Pas (*Scarcity Bottleneck*)
- **Concept** : Limiter le nombre de repas extractibles par pas : au plus $k < N$ agents peuvent réussir à consommer par pas de temps (ex. $k = 2$ ou $3$ repas pour $N = 6$ agents). Si plus de $k$ agents exploitent, le succès est alloué stochastiquement ou échoue.
- **Hypothèse** : La surconsommation simultanée crée une interférence négative directe et immédiate sur le bien-être d'autrui au même pas temporel, restaurant la pression pour l'abstention sans altérer la reward homéostatique standard (`homeostatic`).
- **Paramètres** : `meals_per_step` $k \in \{2, 3, 4, 6\}$.
- **Métriques** : Taux de collisions/conflits, émergence de tours de rôle (*turn-taking*).
- **Ancrage Littérature** : Lerer & Peysakhovich (2017), McKee et al. (2020).

#### Exp A.3 — Alignement Homéostatique de l'Émotion Émise
- **Concept** : Rendre la valence émotionnelle émise $e_i$ cohérente avec la satisfaction personnelle homéostatique : $e_i$ culmine à la satiété optimale $h_i = 0{,}5$ et chute pour un gavage excessif ($h_i > 0{,}5$, inconfort/indigestion) ou la famine ($h_i < 0{,}5$).
- **Hypothèse** : Un agent empathique n'est plus incité à « suralimenter » ses pairs au-delà de leur seuil homéostatique, alignant observation sociale et dynamique de consommation durable.
- **Paramètres** : Émotion quadratique ou gaussienne centrée en $h^* = 0{,}5$ vs linéaire actuelle.
- **Métriques** : Distribution stationnaire des taux de repas des agents, indice de Gini.
- **Ancrage Littérature** : Keramati & Gutkin (2014, *Homeostatic reinforcement learning*), Joffily & Coricelli (2013).

---

### Axe B — Mécanismes Cognitifs & Neurosciences de l'Empathie
*Objectif : Tester des architectures computationnelles fidèles aux dissociations connues en neurosciences affectives.*

#### Exp B.1 — Empathie Ciblée vs Moyennée (*Rawlsian Min-Max vs Distress Weighting*)
- **Concept** : Remplacer l'émotion empathique moyenne $\tilde{e}_i = \frac{1}{N-1} \sum_{j \neq i} e_j$ par une pondération asymétrique donnant priorité aux agents en détresse (famine) :
  $$\tilde{e}_i = \sum_{j \neq i} w_j e_j \quad \text{avec} \quad w_j = \frac{\exp(-\lambda e_j)}{\sum_k \exp(-\lambda e_k)}$$
  ($\lambda = 0$ : moyenne classique ; $\lambda \to \infty$ : focalisation stricte sur le plus mal loti / Maximin rawlsien).
- **Hypothèse** : La pondération sur la détresse brise la constante empathique totale de l'épisode et supprime l'inégalité (réduit drastiquement le Gini) sans nécessiter de bonus de retenue ad-hoc.
- **Paramètres** : $\lambda \in \{0, 1, 3, 10\}$.
- **Métriques** : Gini des repas, durée de vie de la ressource, variance inter-agents de satiété.
- **Ancrage Littérature** : Batson (2011, *Empathy-Altruism Hypothesis*), Decety (2015), Rawls (1971).

#### Exp B.2 — Découplage Perception-Motivation (Vraie Valeur vs Valeur Perçue)
- **Concept** : Dans le code actuel, l'agent observe $x \sim \text{Beta}(\rho)$, mais sa reward empathique utilise la vraie émotion sans bruit. Tester le régime unifié où la reward empathique est calculée sur le signal effectivement perçu $\hat{x}_j$.
- **Hypothèse** : À faible précision ($\rho \to 0$), la reward sociale devient un bruit blanc, désactivant spontanément l'empathie comportementale même avec un $\alpha$ nominal élevé. Cela clarifie le rôle de l'axe 3 (bruit $\rho$).
- **Paramètres** : Condition `reward_on_perceived = True / False` croisée avec $\rho \in \{0, 0.25, 0.5, 0.75, 1.0\}$.
- **Métriques** : Sensibilité de $\Delta \text{steps}$ et Gini à $\rho$.
- **Ancrage Littérature** : de Melo et al. (2014), Preston & de Waal (2002, PAM model).

#### Exp B.3 — Horizon Temporel Asymétrique ($\gamma_o$ Myope vs Hyperbolique)
- **Concept** : Évaluer l'Exp7 sous l'angle de la détresse affective immédiate vs projection cognitive future. Comparer $\gamma_o \in \{0, 0.5, 0.8\}$ (empathie viscérale/affective réactive) à $\gamma_o \in \{0.99, 0.999\}$ (empathie cognitive/altruisme pour les générations futures).
- **Hypothèse** : Une empathie myope ($\gamma_o < \gamma_s$) suffit à stabiliser la coopération si les signaux affectifs reflètent la détresse immédiate, contredisant l'idée reçue qu'un discount élevé est indispensable à la durabilité.
- **Paramètres** : $\gamma_s = 0.99$, $\gamma_o \in \{0.0, 0.5, 0.8, 0.95, 0.99, 0.999\}$.
- **Métriques** : Durée de vie, convergence DQN à deux têtes, stabilité des Q-values.
- **Ancrage Littérature** : Osiński & Karbowski (2017), van Rijn et al. (2024), Fedus et al. (2019).

---

### Axe C — Écologie Sociale : Hétérogénéité & "Tragédie de l'Empathie"
*Objectif : Comprendre la dynamique de groupe quand l'empathie n'est pas uniforme.*

#### Exp C.1 — Populations Mixtes & Parasitisme Social (*Free-Rider Exploitation*)
- **Concept** : Introduire des proportions variables d'agents égoïstes ($\alpha = 0$) et empathiques ($\alpha = 0{,}5$) au sein des 6 agents (ex. 1/5, 2/4, 3/3, 5/1).
- **Hypothèse** : « L'altruisme pathologique » : les agents empathiques se restreignent pour protéger le groupe, mais les agents égoïstes surconsomment sans sanction, captant tous les repas et exacerbant l'inégalité de façon extrême.
- **Paramètres** : Ratio d'agents empathiques $N_{\alpha=0.5} / N \in \{0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6\}$.
- **Métriques** : Gini intra-groupe vs inter-types, ratio de consommation $\text{meals}_{ego} / \text{meals}_{emp}$.
- **Ancrage Littérature** : Oakley et al. (2011, *Pathological Altruism*), Fehr & Fischbacher (2003).

#### Exp C.2 — Explication du "Fairness Reversal" par Asymétrie d'Apprentissage
- **Concept** : Tester si l'inégalité émergente entre agents empathiques identiques provient d'une divergence stochastique précoce (symmetry breaking). Manipuler l'initialisation des poids du DQN (poids identiques vs aléatoires indépendants) et le taux d'exploration $\epsilon$.
- **Hypothèse** : Dès qu'un agent consomme légèrement plus au début, sa satisfaction accrue incite les autres agents empathiques à s'abstenir pour maintenir cette satisfaction, enfermant le système dans une trappe d'inégalité auto-renforcée.
- **Paramètres** : Poids partagés initialement vs découplés ; décroissance $\epsilon$ rapide vs lente.
- **Métriques** : Divergence précoce des politiques (épisodes -1 à 20 via `--eval-dense-first`), Gini final.
- **Ancrage Littérature** : Batson et al. (1995, *Immorality from empathy*), Hughes et al. (2018).

#### Exp C.3 — Distance Sociale & Altruisme Paroissial (*In-Group vs Out-Group*)
- **Concept** : Diviser les 6 agents en 2 groupes de 3. Le vecteur d'observation et la reward empathique appliquent une matrice de proximité sociale $A_{ij} \in [\alpha_{out}, \alpha_{in}]$.
- **Hypothèse** : L'empathie préférentielle envers les proches favorise la coopération locale mais intensifie la tragédie des communs globale par compétition inter-groupes.
- **Paramètres** : $\alpha_{in} = 0.5$, $\alpha_{out} \in \{0.0, 0.1, 0.5\}$.
- **Métriques** : Durée de vie globale, Gini intra-groupe vs inter-groupe.
- **Ancrage Littérature** : Jones & Rachlin (2006, *Social discounting*), De Dreu et al. (2010).

---

### Axe D — Canaux d'Observation & Robustesse du Signal
*Objectif : Généraliser l'étude des contraintes informationnelles au-delà des émotions.*

#### Exp D.1 — Observation Imparfaite de la Ressource ($\rho_{res}$)
- **Concept** : Appliquer l'opérateur `perceive_bounded` au slot de ressource $r_t / r_0$ avec une précision $\rho_{res} \in [0, 1]$, symétrique à la précision émotionnelle $\rho_{emo}$.
- **Hypothèse** : Lorsque l'état écologique est incertain ($\rho_{res} < 1$), le signal social émotionnel reprend un rôle de premier plan (substitut d'information d'état), annulant la prédominance de la ressource constatée dans l'Exp1.
- **Paramètres** : 2×2 factoriel $\rho_{res} \in \{0.2, 1.0\} \times \rho_{emo} \in \{0.0, 1.0\}$.
- **Métriques** : Interaction $\rho_{res} \times \rho_{emo}$ sur la soutenabilité (ANOVA).
- **Ancrage Littérature** : Ostrom (1990, principe de surveillance/monitoring), Janssen (2022).

#### Exp D.2 — Signaux Émotionnels Stratégiques / Trompeurs (*Cheap Talk & Honesty*)
- **Concept** : Autoriser les agents à découpler leur état interne réel de l'émotion diffusée (ex. afficher une détresse feinte $e_{fake} < e_{true}$ pour inciter les autres à la retenue).
- **Hypothèse** : En l'absence de coût de signalement (signalling cost), le canal émotionnel s'effondre en babillage sans valeur (*cheap talk babbling equilibrium*) et la soutenabilité régresse.
- **Paramètres** : Émission honnête vs émission apprise / trompeuse.
- **Métriques** : Corrélation signal-satiété, soutenabilité de la ressource.
- **Ancrage Littérature** : Crawford & Sobel (1982), de Melo et al. (2019), Zahavi (1975).

---

### Axe E — Dépendance à la Dynamique Environnementale & Optimalité Hors-Distribution (*Frozen Policy Transfer*)
*Objectif : Exploiter les poids sauvegardés (`<run>_weights.pt`) pour tester si la politique apprise est strictement dépendante de la trajectoire de déclin inéluctable ou si elle a acquis une règle de régulation fonctionnelle transposable à un régime renouvelable.*

#### Exp E.1 — Politiques Gelées sur Ressource Renouvelable : Découplage & Efficacité Écologique
- **Concept** : Reprendre les agents en fin d'apprentissage avec leurs poids **strictement gelés** (aucun réapprentissage, $\epsilon = 0$, évaluation greedy pure) et les exécuter dans une condition où la ressource se régénère ($R_{t+1} = \min(K, R_t - \sum C_i + \text{regen})$).
- **Hypothèse scientifique** :
  - *Sur-apprentissage dynamique (Overfitting aux transitions)* : Dans le modèle actuel, $r_t / r_0$ décroît de façon monotone et agit comme une horloge temporelle de fin d'épisode. Si les agents ont sur-appris cette trajectoire descendante, leur politique échoue face à une ressource renouvelable (surconsommation aveugle ou blocage inadapté).
  - *Régulation fonctionnelle & Heuristiques sociales* : Si l'empathie ($\alpha > 0$) a fait émerger une règle de rétroaction robuste ($s \mapsto a$) régulée par les émotions d'autrui, les agents stabilisent spontanément la ressource autour d'un point d'équilibre pérenne.
- **Niveaux d'optimalité mesurés** :
  - **Efficacité écologique ($\eta_{MSY}$)** : Ratio entre les repas totaux récoltés par la politique gelée et le rendement maximal soutenable théorique (**Maximum Sustainable Yield**) sur $T = 1000$ pas.
  - **Taux de capture vs Seuil d'effondrement** : Maintien de la ressource au-dessus du seuil critique d'extinction.
  - **Dépendance à la dynamique selon le canal d'observation** : Comparer C/D (avec émotions) vs A/B (aveugles). Le signal affectif rend-il la politique moins dépendante de la dynamique brute de la ressource ?
- **Ancrage Littérature** :
  - Zhang et al. (2018) : *A Study on Overfitting in Deep Reinforcement Learning*.
  - Kirk et al. (2023, *JAIR*) : *A Survey of Zero-shot Generalisation in Deep Reinforcement Learning* (robustesse aux décalages de dynamique $P(s' \mid s, a)$).
  - Rand et al. (2014, *Nature Communications*) : *Social Heuristics Hypothesis*.
  - Gigerenzer & Todd (1999) : *Ecological Rationality*.

---

## 3. Matrice de Priorisation & Feuille de Route pour le Papier Journal

| Expérience | Question Scientifique Centrale | Complexité Code | Priorité Journal | Complémentarité avec le Plan Actuel |
|---|---|---|---|---|
| **E.1 (Politiques Gelées / Optimalité CPR)** | L'apprentissage dépend-il de la trajectoire de déclin ou régule-t-il la ressource renouvelable ? | Très faible (utilise `<run>_weights.pt` + boucle eval) | ⭐⭐⭐ **Haute** | **Pépite conceptuelle** : valorise immédiatement les poids sauvegardés sans nouveau training lourd. |
| **A.1 (CPR Logistique)** | La retenue émerge-t-elle sans bonus codé si la ressource est renouvelable ? | Moyenne (`env.step`) | ⭐⭐⭐ **Haute** | Résout la critique majeure de la retenue ad-hoc (§2.2 du docx). |
| **B.1 (Rawlsian Min-Max)** | Prioriser les agents affamés supprime-t-il le renversement d'équité ? | Faible (`SocialRewardCalculator`) | ⭐⭐⭐ **Haute** | Réponse directe à Hughes 2018 (R2) sur l'inégalité. |
| **C.1 (Populations Mixtes)** | L'empathie est-elle exploitée par les agents égoïstes (parasitisme) ? | Nulle (déjà supporté par `agent_configs`) | ⭐⭐⭐ **Haute** | Donne une dimension sociologique / évolutionniste forte au papier. |
| **B.3 (Social Discount $\gamma_o$)** | L'empathie affective requiert-elle un horizon long ou court ? | Faible (Exp7 déjà codée) | ⭐⭐⭐ **Haute** | Consolide l'Axe 2 (Motivation : intensité $\alpha$ vs horizon $\gamma_o$). |
| **C.2 (Divergence Précoce)** | L'inégalité découle-t-elle d'une rupture de symétrie à l'initialisation ? | Faible (exploite `--eval-dense-first`) | ⭐⭐ **Moyenne** | Apporte une analyse fine des poids DQN (prévue au plan). |
| **D.1 (Bruit Ressource $\rho_{res}$)** | L'émotion compense-t-elle une observation floue de la ressource ? | Faible (`perceive_bounded` sur slot 0) | ⭐⭐ **Moyenne** | Enrichit l'Axe 3 (Observation) avec un pont solide vers Ostrom. |
| **A.3 (Émotion Cohérente)** | L'alignement homéostatique de l'émotion réduit-il la surconsommation ? | Faible (`emotion_from_rate`) | ⭐⭐ **Moyenne** | Répare l'incohérence conceptuelle identifiée en réunion. |
| **D.2 (Signaux Stratégiques)** | L'empathie résiste-t-elle au mensonge émotionnel ? | Haute (espace d'action étendu) | ⭐ **Future** | Piste passionnante pour une suite / journal ultérieur. |

---

## 4. Recommandation Immédiate pour la Réunion avec Ismael

1. **Lancer le Test de Dépendance à la Dynamique (Exp E.1)** : Les poids finaux de l'Axe 1 étant déjà sauvegardés (`<run>_weights.pt`), exécuter une simple évaluation greedy sans apprentissage dans un environnement renouvelable pour déterminer à quel niveau d'optimalité (MSY) les politiques se situent hors de leur dynamique d'entraînement.
2. **Valider l'Expérience B.1 (Empathie Rawlsienne)** : Elle ne modifie pas l'environnement et règle simultanément le problème du bonus de retenue codé et l'inégalité de Gini, en restant strictement dans le paradigme affectif.
3. **Tester l'Expérience C.1 (Populations Mixtes)** dès maintenant avec le code existant : `main_gt_ais.py` permet déjà d'assigner des $\alpha$ hétérogènes par agent sans modifier une seule ligne du simulateur.
4. **Activer la campagne Exp7 ($\gamma_o$)** en intégrant les valeurs myopes ($\gamma_o < \gamma_s$) pour défendre la nature affective et non purement utilitariste du signal émotionnel.
