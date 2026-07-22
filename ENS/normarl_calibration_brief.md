# Brief de calibration analytique — NORMARL

## Contexte

On part du papier **NORMARL** (Shiravand et al. 2025) qui simule des agents
apprenant des normes sociales pour gérer une ressource commune. On a développé
une version analytique simplifiée du modèle et on cherche à calibrer deux
paramètres libres $a$ et $b$ sur les résultats de leurs simulations.

---

## Le modèle analytique

**Dynamique de la ressource :**

$$E_{t+1} = (1+\delta)E_t - N\bar{q}_t$$

À l'équilibre : $\delta E^* = N\bar{q}^*$

**Hypothèses simplificatrices :**

- Chaque agent a probabilité $1/2$ d'être over-consumer (consomme au-dessus
  de la norme $\bar{q}$) et $1/2$ d'être under-consumer
- On fusionne belief et vraie consommation
- On ignore la softmax : chaque agent optimise directement sa valeur

**Action optimale :**

- Under-consumer (pas de coût social) : $Q^*_{under} = \dfrac{1}{4}$
- Over-consumer : $Q^*_{over} = \dfrac{1}{4(1+\lambda)^2}$ où $\lambda = \beta(a - bE^*)$

**Consommation moyenne à l'équilibre :**

$$\bar{q} = \frac{1}{8}\left(1 + \frac{1}{(1+\lambda)^2}\right)$$

**Équation d'équilibre implicite en $E^*$ :**

$$\delta E^* = \frac{N}{8}\left(1 + \frac{1}{(1+\beta(a-bE^*))^2}\right)$$

**Réécriture en cubique** via $u = 1 + \beta(a - bE^*)$ :

$$-\frac{\delta}{b\beta}u^3 + \left(\frac{\delta}{b}\left(a + \frac{1}{\beta}\right) - \frac{N}{8}\right)u^2 - \frac{N}{8} = 0$$

La bifurcation $\beta_c$ est la valeur où cette cubique admet une racine
double, i.e. $f(u^*) = 0$ et $f'(u^*) = 0$ simultanément.

---

## Paramètres fixes (donnés dans le papier)

```python
N     = 100      # nombre d'agents
delta = 0.02     # taux de régénération
E0    = 1000     # stock initial
```

---

## Paramètres libres à calibrer

```python
a   # intensité du coût social (>0)
b   # sensibilité du coût social à E (>0)
```

Le coût social d'un over-consumer est :

$$S_i(Q) = \beta(Q - \bar{q}) \cdot \max\{0,\, a - bE\}$$

---

## Cibles de calibration

Lues depuis la Figure 2a du papier (valeurs asymptotiques de $E_t$) :

```python
targets = {
    'beta_c' : 0.62,   # bifurcation entre déplétion et stabilité
    'E_b065' : 700,    # E*(β=0.65)
    'E_b075' : 850,    # E*(β=0.75)
    'E_b100' : 950,    # E*(β=1.00)
}
```

---

## Ce qu'on sait déjà

- Le solveur par bisection sur l'équation implicite fonctionne bien pour
  trouver $E^*$ étant donnés $(a, b, \beta)$
- $\beta_c$ se trouve par scan sur $\beta$ jusqu'à la première valeur où $E^* > 0$
- La contrainte $E^*(\beta=1.0) = 950$ est **incompatible** avec l'hypothèse
  50/50 si $b$ est trop petit — parce que le plancher théorique
  $E^*_{\beta\to\infty} = N/8\delta = 625$ est inférieur à 950. Donc soit
  les cibles en valeur absolue de $E^*$ sont à prendre avec souplesse
  (tolérance large), soit il faut relâcher l'hypothèse 50/50 et paramétrer
  $p(\beta) \in (0, 0.5]$
- Le résultat le plus robuste est $\beta_c \approx 0.62$ — c'est la
  contrainte principale

---

## Ce que le code doit faire

1. Implémenter le solveur bisection pour $E^*(a, b, \beta)$
2. Implémenter le calcul de $\beta_c(a, b)$ par scan
3. Faire un **grid search** sur $(a, b)$ avec $a \in [0.5, 5]$ et
   $b \in [10^{-4}, 10^{-1}]$ (échelle log pour $b$)
4. Définir une **fonction de coût** pondérée — en priorité $\beta_c$,
   secondairement les valeurs de $E^*$
5. Affiner avec une optimisation locale (Nelder-Mead ou `scipy.optimize.minimize`)
6. Produire un graphe $E^*(\beta)$ avec les paramètres optimaux superposé
   aux cibles

---

## Contrainte structurelle importante

Si les cibles $E^*$ absolues sont inaccessibles avec l'hypothèse 50/50, le
code doit le détecter et reporter clairement :

> **"Cibles E\* incompatibles avec le plancher N/8δ = 625"**

Dans ce cas, calibrer uniquement sur $\beta_c$ en laissant $E^*$ libre, et
reporter l'écart résiduel.
