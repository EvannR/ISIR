"""
Calibration analytique du modele NORMARL (Shiravand et al. 2025).

Modele simplifie (sans softmax, hypothese 50/50 over/under-consumers):
  Q*_under = 1/4
  Q*_over  = 1/(4*(1+lambda)^2)   avec lambda = beta*max(0, a - b*E)
  q_bar    = (1/8)*(1 + 1/(1+lambda)^2)
  Equilibre: delta*E* = N*q_bar(E*)

Physique cle:
  - Pour E > a/b : pas de cout social, q_bar = 1/4, racine a E=1250 (REPULSEUR)
  - Pour E < a/b : cout social actif, q_bar < 1/4, bifurcation selle-noeud
  - L'equilibre STABLE E*_2 est dans (0, a/b), trouve par scan + bisection
  - beta_c est la bifurcation selle-noeud (max de f touche 0)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script runs
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq

# ── Parametres fixes ────────────────────────────────────────────────────────
N     = 100
DELTA = 0.02
E0    = 1000
E_FLOOR = N / (8 * DELTA)   # 625 : plancher theorique quand beta->inf

# ── Cibles lues sur Fig. 2a ─────────────────────────────────────────────────
TARGETS = {
    'beta_c' : 0.62,
    'E_b065' : 700.0,
    'E_b075' : 850.0,
    'E_b100' : 950.0,
}

BETAS_TARGET = {0.65: TARGETS['E_b065'],
                0.75: TARGETS['E_b075'],
                1.00: TARGETS['E_b100']}


# ── Residuel d'equilibre ────────────────────────────────────────────────────

def q_bar_given_E(E, a, b, beta):
    lam = beta * max(0.0, a - b * E)
    return (1.0 / 8.0) * (1.0 + 1.0 / (1.0 + lam) ** 2)


def f_eq(E, a, b, beta):
    """f(E) = delta*E - N*q_bar(E). Stable equilibrium: f crosses 0 from + to -."""
    return DELTA * E - N * q_bar_given_E(E, a, b, beta)


# ── Localisation analytique du maximum de f ─────────────────────────────────

def E_at_fmax(a, b, beta):
    """
    f(E) est concave sur (0, a/b). Son maximum est en f'(E)=0 :
      delta = N*beta*b / (4*u^3)  =>  u^3 = N*beta*b/(4*delta)
      E_max = (a - (u-1)/beta) / b
    """
    u3 = N * beta * b / (4.0 * DELTA)
    u = u3 ** (1.0 / 3.0)
    E_m = (a - (u - 1.0) / beta) / b
    E_thresh = a / b
    return float(np.clip(E_m, 0.1, E_thresh - 0.01))


# ── Solveur : equilibre STABLE dans (E_max, a/b) ────────────────────────────

def solve_Estar_stable(a, b, beta):
    """
    L'equilibre stable E*_2 est dans (E_max, a/b) ou f est monotone decroissante.
    f(E_max) > 0 ssi beta > beta_c.
    Utilise brentq directement — O(log n), pas de scan.
    """
    if b <= 1e-10 or a <= 0:
        return 0.0
    E_thresh = a / b
    E_m = E_at_fmax(a, b, beta)
    if f_eq(E_m, a, b, beta) <= 0:
        return 0.0                    # beta < beta_c
    f_hi = f_eq(E_thresh - 0.01, a, b, beta)
    if f_hi >= 0:
        return 0.0                    # pas de croisement (E* > a/b, regime lineaire)
    try:
        return brentq(f_eq, E_m, E_thresh - 0.01, args=(a, b, beta), xtol=1e-4)
    except ValueError:
        return 0.0


# ── Calcul analytique de beta_c (condition double-racine) ────────────────────

def compute_beta_c(a, b):
    """
    A la bifurcation selle-noeud : f(E*_c)=0 ET f'(E*_c)=0.
    Apres substitution : 3/(2u^2) - 1/u^3 = K - 0.5, K = 4*a*delta/(N*b)
    Solution par brentq sur u > 1, puis beta_c = u^3 / (N*b/(4*delta)).
    """
    if b <= 1e-10 or a <= 0:
        return np.nan
    C = N * b / (4.0 * DELTA)      # = N*b/(4*delta)
    K = a / C                       # = 4*a*delta/(N*b)
    rhs = K - 0.5
    if rhs <= 0 or rhs >= 0.5:
        # No bifurcation in this regime; fall back to scan
        for beta in np.linspace(0.01, 5.0, 200):
            if solve_Estar_stable(a, b, beta) > 0:
                return beta
        return np.nan

    def eq(u):
        return 3.0 / (2.0 * u * u) - 1.0 / (u * u * u) - rhs

    try:
        u_c = brentq(eq, 1.0 + 1e-6, 200.0)
        return u_c ** 3 / C
    except ValueError:
        return np.nan


# ── Fonction de cout ─────────────────────────────────────────────────────────

def cost_function(params, w_betac=10.0, w_Estar=1.0):
    a, b = params
    if a <= 0.1 or b <= 1e-5:
        return 1e9

    beta_c_pred = compute_beta_c(a, b)
    if np.isnan(beta_c_pred):
        return 1e9

    cost = w_betac * (beta_c_pred - TARGETS['beta_c']) ** 2

    for beta_val, E_target in BETAS_TARGET.items():
        E_pred = solve_Estar_stable(a, b, beta_val)
        if E_pred > 0:
            cost += w_Estar * ((E_pred - E_target) / E_target) ** 2
        else:
            cost += w_Estar * 1.0  # penalite depletion

    return cost


# ── Grid search ──────────────────────────────────────────────────────────────

def grid_search(n_a=25, n_b=25):
    # Domaine : a/b doit couvrir E0=1000, donc a/b > 1000
    # Valeurs analytiques attendues : a~6-8, b~0.005-0.010
    a_vals = np.linspace(1.0, 12.0, n_a)
    b_vals = np.logspace(-3, -1.5, n_b)   # ~0.001 a 0.032

    best_cost = np.inf
    best_params = (None, None)
    cost_grid = np.full((n_a, n_b), np.nan)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            c = cost_function((a, b))
            cost_grid[i, j] = c
            if c < best_cost:
                best_cost = c
                best_params = (a, b)
        if (i + 1) % 5 == 0:
            print(f"  grid: {i+1}/{n_a} lignes traitees, best so far: {best_cost:.4f}")

    return best_params, best_cost, cost_grid, a_vals, b_vals


# ── Optimisation locale (log-espace pour b) ───────────────────────────────────

def cost_log(log_params):
    a, log_b = log_params
    b = np.exp(log_b)
    return cost_function((a, b))


def local_optimize(a0, b0):
    result = minimize(
        cost_log,
        x0=[a0, np.log(b0)],
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 10000},
    )
    a_opt = result.x[0]
    b_opt = np.exp(result.x[1])
    return (a_opt, b_opt), result.fun


# ── Verification plancher ─────────────────────────────────────────────────────

def check_floor():
    print(f"\nPlancher theorique N/8*delta = {E_FLOOR:.0f}")
    all_ok = True
    for beta_v, E_t in BETAS_TARGET.items():
        if E_FLOOR >= E_t:
            print(f"  [!] E*(beta={beta_v}) = {E_t} <= plancher {E_FLOOR:.0f} : INCOMPATIBLE")
            all_ok = False
        else:
            print(f"  [ok] E*(beta={beta_v}) = {E_t} > plancher {E_FLOOR:.0f}")
    return all_ok


# ── Verification analytique (a,b depuis beta_c + E*(beta=1)) ─────────────────

def analytical_estimate():
    """
    Derivation analytique de (a, b) depuis :
      - beta_c = 0.62 (bifurcation selle-noeud)
      - E*(beta=1.0) = 950
    Utile pour initialiser la recherche.
    """
    # De la bifurcation selle-noeud + E*(beta=1)=950, on derive u_c via:
    # 0.260*u_c^3 - 1.5*u_c + 1.24 = 0  (voir bloc-notes)
    beta_c = TARGETS['beta_c']
    E_950  = 950.0
    beta_1 = 1.0

    # q_bar a E*=950, beta=1 -> lambda_1
    q_target = DELTA * E_950 / N   # = 0.19
    val = 8 * q_target - 1         # = 0.52
    lam_1 = 1.0 / np.sqrt(val) - 1  # = 0.387

    # a - 950*b = lam_1 / beta_1 = 0.387
    # Bifurcation: (1+lam_c)^3 = N*beta_c*b / (4*DELTA)
    # Resoudre le systeme en u_c = 1+lam_c par polynome cubique
    # 0.260*u_c^3 - 1.5*u_c + 1.24 = 0  (approximation numerique)
    # Solution non-triviale ~1.74
    C = N * beta_c / (4 * DELTA)   # = 775
    A = lam_1 / beta_1             # = a - E_950*b

    def poly_uc(uc):
        b_uc = uc**3 / C
        a_uc = uc * (1.5 / beta_c) - 1.0 / beta_c + 0.5 * uc**3 / (beta_c * C * beta_c)
        # Simplified: use a - E_950*b = A
        a_uc2 = (1.5 * uc - 1 + 0.5 * uc**3) / beta_c
        b_uc2 = uc**3 / C
        return a_uc2 - E_950 * b_uc2 - A

    try:
        uc_opt = brentq(poly_uc, 1.01, 5.0)
        b_est = uc_opt**3 / C
        a_est = (1.5 * uc_opt - 1 + 0.5 * uc_opt**3) / beta_c
        print(f"\n  Estimation analytique : a={a_est:.3f}, b={b_est:.5f}")
        print(f"  (u_c = {uc_opt:.3f}, E*_c = {625*(1+1/uc_opt**2):.0f})")
        return a_est, b_est
    except Exception as e:
        print(f"\n  Estimation analytique impossible : {e}")
        return 7.0, 0.007


# ── Graphe E*(beta) ──────────────────────────────────────────────────────────

def plot_Estar_curve(a_opt, b_opt, save_path='ENS/Estar_calibration.png'):
    beta_grid = np.linspace(0.3, 2.0, 400)
    E_pred = [solve_Estar_stable(a_opt, b_opt, beta) for beta in beta_grid]

    beta_c_pred = compute_beta_c(a_opt, b_opt)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(beta_grid, E_pred, 'b-', lw=2,
            label=f'E*(b) fitted  a={a_opt:.3f}, b={b_opt:.5f}')
    ax.scatter(list(BETAS_TARGET.keys()), list(BETAS_TARGET.values()),
               color='red', s=80, zorder=5, label='Targets Fig. 2a')
    ax.axvline(TARGETS['beta_c'], color='orange', ls='--',
               label=f"beta_c target = {TARGETS['beta_c']}")
    ax.axhline(E_FLOOR, color='gray', ls=':',
               label=f'Floor N/8*delta = {E_FLOOR:.0f}')
    if not np.isnan(beta_c_pred):
        ax.axvline(beta_c_pred, color='steelblue', ls='--', alpha=0.7,
                   label=f'beta_c predicted = {beta_c_pred:.3f}')
    ax.set_xlabel('beta (norm internalization)')
    ax.set_ylabel('E* (equilibrium stock)')
    ax.set_title('NORMARL calibration  E*(beta)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1100)
    ax.set_xlim(0.3, 2.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Graph saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 0. Verification plancher
    check_floor()

    # 1. Estimation analytique (initialisation)
    a_init, b_init = analytical_estimate()

    # 2. Grid search
    print("\nGrid search...")
    (a0, b0), cost0, cost_grid, a_vals, b_vals = grid_search(n_a=25, n_b=25)
    print(f"  Best grid : a={a0:.3f}, b={b0:.5f}  (cost={cost0:.4f})")

    # Si le grid est loin de l'estimation analytique, forcer l'init analytique
    cost_ana = cost_function((a_init, b_init))
    print(f"  Analytical init cost : {cost_ana:.4f}")
    if cost_ana < cost0:
        print("  -> Using analytical estimate as starting point")
        a0, b0 = a_init, b_init

    # 3. Optimisation locale
    print("\nNelder-Mead optimisation...")
    (a_opt, b_opt), cost_opt = local_optimize(a0, b0)
    print(f"  Optimum : a={a_opt:.4f}, b={b_opt:.6f}  (cost={cost_opt:.6f})")

    # 4. Rapport
    beta_c_pred = compute_beta_c(a_opt, b_opt)
    print(f"\n-- Results --")
    print(f"  a = {a_opt:.4f},  b = {b_opt:.6f}")
    print(f"  a/b = {a_opt/b_opt:.1f}  (social cost active for E < {a_opt/b_opt:.0f})")
    print(f"  beta_c predicted = {beta_c_pred:.4f}  (target {TARGETS['beta_c']})")
    for beta_v, E_t in BETAS_TARGET.items():
        E_p = solve_Estar_stable(a_opt, b_opt, beta_v)
        print(f"  E*(beta={beta_v:.2f}) = {E_p:.1f}  (target {E_t:.0f})")

    # 5. Graphe
    plot_Estar_curve(a_opt, b_opt)
