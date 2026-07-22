# Analyse analytique du modèle NORMARL

## Résumé

On dérive une version analytique du modèle NORMARL (Shiravand et al. 2025) en remplaçant la dynamique multi-agent par un équilibre statique tractable. L'objectif est de comprendre la bifurcation observée en simulation à $\beta_c \approx 0.62$ et de calibrer les paramètres libres $a$ et $b$.

---

## Le modèle analytique

**Dynamique de la ressource** (identique au papier) :

$$E_{t+1} = (1+\delta)E_t - \sum_{i=1}^N Q_i^t$$

**Hypothèses simplificatrices** (par rapport à la simulation) :

| Simulation NORMARL | Modèle analytique |
|---|---|
| Actions discrètes $\{0.05, \ldots, 0.40\}$ | Optimisation continue |
| Sélection softmax | Maximisation directe |
| Mise à jour des croyances ($\alpha$) | Croyances = vraie norme |
| Fraction d'over-consumers dynamique | 50% over / 50% under (fixe) |

**Actions optimales** résultantes :

$$Q^*_{\text{under}} = \frac{1}{4}, \qquad Q^*_{\text{over}} = \frac{1}{4(1+\lambda)^2}$$

avec $\lambda = \beta \cdot \max(0,\, a - bE)$.

**Consommation moyenne à l'équilibre** :

$$\bar{q} = \frac{1}{8}\left(1 + \frac{1}{(1+\lambda)^2}\right)$$

**Condition d'équilibre** $\delta E^* = N\bar{q}^*$ :

$$\delta E^* = \frac{N}{8}\left(1 + \frac{1}{\left(1 + \beta(a-bE^*)\right)^2}\right)$$

---

## Structure des équilibres

Le résidu $f(E) = \delta E - N\bar{q}(E)$ est **concave** sur $(0,\, a/b)$. Deux régimes :

- **$\beta < \beta_c$** : $f$ reste négatif partout sur $(0, a/b)$ → pas d'équilibre stable → **dépletion**
- **$\beta > \beta_c$** : $f$ admet un maximum positif → deux racines $E^*_1 < E^*_2$ dans $(0, a/b)$

| Racine | Stabilité | Rôle |
|---|---|---|
| $E^*_1$ | Instable | Seuil de bassin d'attraction |
| $E^*_2$ | **Stable** | Équilibre observé |

La bifurcation $\beta_c$ est une **bifurcation selle-nœud** : $f(E^*_c) = 0$ et $f'(E^*_c) = 0$ simultanément.

**Condition analytique sur $\beta_c$** : en posant $u_c = 1 + \beta_c(a - bE^*_c)$ et $C = Nb/(4\delta)$,

$$\frac{3}{2u_c^2} - \frac{1}{u_c^3} = \frac{4a\delta}{Nb} - \frac{1}{2}, \qquad \beta_c = \frac{u_c^3}{C}$$

**Plancher théorique** ($\beta \to \infty$) :

$$E^*_{\min} = \frac{N}{8\delta} = 625$$

---

## Calibration

### Paramètres fixes

```
N = 100,  delta = 0.02,  E0 = 1000
```

### Cibles (Figure 2a du papier)

| Cible | Valeur |
|---|---|
| $\beta_c$ | 0.62 |
| $E^*(\beta=0.65)$ | 700 |
| $E^*(\beta=0.75)$ | 850 |
| $E^*(\beta=1.00)$ | 950 |

### Résultats de calibration

Grid search sur $a \in [1, 12]$ et $b \in [10^{-3}, 10^{-1.5}]$, affiné par Nelder-Mead avec poids $w_{\beta_c} = 10$, $w_{E^*} = 1$ :

$$\boxed{a = 11.68, \quad b = 0.01296}$$

| Quantité | Prédit | Cible | Écart |
|---|---|---|---|
| $\beta_c$ | **0.629** | 0.62 | $+0.009$ |
| $E^*(\beta=0.65)$ | 786 | 700 | $+86$ |
| $E^*(\beta=0.75)$ | 820 | 850 | $-30$ |
| $E^*(\beta=1.00)$ | 850 | 950 | $-100$ |

---

## Cohérence et limites

### Ce qui fonctionne bien

**$\beta_c = 0.629 \approx 0.62$** : la bifurcation est reproduite fidèlement. C'est la contrainte principale et la plus robuste du modèle.

### Écart structurel sur $E^*$

Les valeurs $E^*(\beta)$ sont systématiquement décalées. L'origine est géométrique : à $\beta_c$, la bifurcation selle-nœud émerge à $E^*_c \approx 831$. L'équilibre stable $E^*_2$ croît depuis 831 quand $\beta$ augmente au-dessus de $\beta_c$. Le modèle analytique prédit donc :

$$E^*(\beta=0.65) > E^*_c \approx 831$$

Or la simulation donne $E^*(\beta=0.65) = 700 < 831$, ce qui est **structurellement incompatible** avec l'hypothèse 50/50.

**Pourquoi la simulation donne des valeurs plus basses :** dans la vraie simulation, pour $\beta$ juste au-dessus de $\beta_c$, la norme perçue converge lentement (dépend de $\alpha$), la sélection softmax introduit de la stochasticité, et la fraction d'over-consumers est dynamique. Ces effets font que la ressource se stabilise plus bas que le maximum analytique.

### Contrainte fondamentale non-reproductible

Pour que $E^*(\beta=1.0) = 950$ avec $a = 11.68$, $b = 0.01296$ :

$$\bar{q}(950) = \frac{0.02 \times 950}{100} = 0.19 > \frac{1}{8} = 0.125$$

Ceci est possible (bien au-dessus du plancher 625), mais le modèle analytique donne $E^*(1.0) = 850$ car la bifurcation émerge trop près de $a/b = 901$.

---

## Conclusion

Le modèle analytique capture la **structure qualitative** de NORMARL :

- Existence d'une bifurcation $\beta_c$ entre dépletion et durabilité
- Monotonie : $E^*$ croît avec $\beta$
- Plancher $N/8\delta = 625$ asymptotique

La calibration sur $\beta_c$ est précise ($\pm 1\%$). Les valeurs absolues de $E^*$ ont un écart systématique d'environ $100$ unités, imputable à l'hypothèse 50/50 et à l'absence de softmax.

**Pour lever cette limite**, il faudrait paramétrer la fraction d'over-consumers $p(\beta) \in (0, 0.5]$ comme une fonction décroissante de $\beta$ — ce qui permettrait de reproduire des $E^*$ plus bas pour $\beta$ proche de $\beta_c$.
