"""
duffing.py — Duffing oscillator dynamics and local Jacobian estimation.

    H(q,p) = p²/2 + q⁴/4 − q²/2
    qdot = p,   pdot = q − q³

Key idea: perturb z₀ → z₀+ε, integrate for small dt, and use the
linear map  Δ ≈ DΦ_dt · ε  to estimate the flow Jacobian.  Then
    A(z₀) = (DΦ_dt − I) / dt ≈ J ∇²H(z₀)
gives the local Hessian of H, which we smooth with a GP and integrate
twice to reconstruct H.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ─── Hamiltonian and true Hessian ────────────────────────────────────────────

def hamiltonian(q, p):
    return 0.5 * p**2 + 0.25 * q**4 - 0.5 * q**2

def true_hessian(q, p):
    """Return (H_qq, H_qp, H_pp) of the true Duffing Hamiltonian."""
    return 3.0 * q**2 - 1.0, 0.0 * q, 1.0 + 0.0 * q


# ─── Integration ─────────────────────────────────────────────────────────────

def _rhs(t, z):
    return [z[1], z[0] - z[0]**3]

def integrate(z0, T, dt=0.005):
    """Integrate from z0 for time T, return z(T)."""
    sol = solve_ivp(_rhs, [0, T], z0, method='RK45',
                    max_step=dt, rtol=1e-10, atol=1e-12)
    return sol.y[:, -1]

def integrate_trajectory(z0, T, n_save=20, dt=0.005):
    """Return (times, states) with n_save evenly-spaced snapshots."""
    t_eval = np.linspace(0, T, n_save)
    sol = solve_ivp(_rhs, [0, T], z0, method='RK45',
                    t_eval=t_eval, max_step=dt, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T   # times (n_save,), states (n_save, 2)


# ─── Jacobian / Hessian estimation from IC perturbations ────────────────────

def estimate_hessian_field(grid_q, grid_p, dt_jac=0.1,
                           n_eps=50, eps_std=0.02, seed=42):
    """
    At each grid point z₀, estimate ∇²H(z₀) from IC perturbations.

    Algorithm at each z₀:
      1. Draw n_eps perturbations ε ∼ N(0, σ²I)
      2. Integrate z₀ and z₀+ε forward by dt_jac
      3. Δ⁽ⁱ⁾ = Φ(z₀+ε⁽ⁱ⁾) − Φ(z₀)  ≈  DΦ · ε⁽ⁱ⁾
      4. Least-squares for DΦ:  DΦ^T = (E^T E)^{-1} E^T D
         where E = (n, 2) of ε's, D = (n, 2) of Δ's
      5. A = (DΦ − I) / dt  ≈  J ∇²H
      6. ∇²H = −J · A  (since J^{-1} = −J)

    Returns
    -------
    Z_grid : (M, 2)
    Hqq, Hqp, Hpp : (M,) — estimated Hessian components
    """
    rng = np.random.default_rng(seed)
    QQ, PP = np.meshgrid(grid_q, grid_p)
    Z_grid = np.column_stack([QQ.ravel(), PP.ravel()])
    M = len(Z_grid)

    J_mat = np.array([[0, 1], [-1, 0]], dtype=float)

    Hqq = np.zeros(M)
    Hqp = np.zeros(M)
    Hpp = np.zeros(M)
    mean_vel = np.zeros((M, 2))   # mean velocity ≈ J∇H(z₀)

    for i in range(M):
        z0 = Z_grid[i]
        z0_T = integrate(z0, dt_jac)

        # Mean velocity at z₀ (from nominal trajectory, no perturbation)
        mean_vel[i] = (z0_T - z0) / dt_jac

        # Draw perturbations and integrate
        E = rng.normal(0, eps_std, size=(n_eps, 2))
        D = np.zeros((n_eps, 2))
        for j in range(n_eps):
            z_pert_T = integrate(z0 + E[j], dt_jac)
            D[j] = z_pert_T - z0_T

        # Least-squares: D ≈ E @ DPhi^T  →  DPhi^T = pinv(E) @ D
        DPhi_T = np.linalg.lstsq(E, D, rcond=None)[0]   # (2, 2)
        DPhi = DPhi_T.T                                    # (2, 2)

        # Extract A = (DPhi - I) / dt
        A = (DPhi - np.eye(2)) / dt_jac

        # ∇²H = -J @ A  (since A = J ∇²H → ∇²H = J^{-1} A = -J A)
        H2 = -J_mat @ A
        # H2 should be symmetric; average off-diagonals
        Hqq[i] = H2[0, 0]
        Hpp[i] = H2[1, 1]
        Hqp[i] = 0.5 * (H2[0, 1] + H2[1, 0])

    return Z_grid, Hqq, Hqp, Hpp, mean_vel


# ─── Noisy velocity observations for GP kernel comparison ────────────────────

def generate_velocity_data(grid_q, grid_p, n_eps=40, eps_std=0.02, seed=42):
    """
    At each nominal grid point z₀, observe the vector field at z₀+ε
    but record it as being "at z₀".

    Since qdot = p and pdot = q − q³:
      qdot(z₀+ε) = p₀+ε_p           (linear in ε → homoscedastic noise)
      pdot(z₀+ε) = (q₀+ε_q) − (q₀+ε_q)³  (nonlinear → heteroscedastic)

    The pdot noise scales as ε_q(1−3q₀²) to first order, so it's large
    where |1−3q₀²| is large (near the separatrix transition) and small
    deep in the wells.

    Returns
    -------
    Z_in   : (N, 2) — nominal grid points (repeated n_eps times each)
    qdot   : (N,)   — noisy qdot observations
    pdot   : (N,)   — noisy pdot observations
    """
    rng = np.random.default_rng(seed)
    QQ, PP = np.meshgrid(grid_q, grid_p)
    centers = np.column_stack([QQ.ravel(), PP.ravel()])

    Z_in, qdot_list, pdot_list = [], [], []
    for z0 in centers:
        for _ in range(n_eps):
            eps = rng.normal(0, eps_std, size=2)
            z_pert = z0 + eps
            qd = z_pert[1]                                    # qdot = p
            pd = z_pert[0] - z_pert[0]**3                     # pdot = q − q³
            Z_in.append(z0.copy())
            qdot_list.append(qd)
            pdot_list.append(pd)

    return np.array(Z_in), np.array(qdot_list), np.array(pdot_list)

def evolve_ensemble(z0, sigma, n_samples, T, n_save=8, seed=0):
    """
    Draw n_samples perturbations around z0, integrate each forward,
    return snapshots at n_save times.

    Returns
    -------
    times     : (n_save,)
    ensembles : (n_save, n_samples, 2)
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=(n_samples, 2))
    z0s = z0[np.newaxis, :] + eps

    t_eval = np.linspace(0, T, n_save)
    ensembles = np.zeros((n_save, n_samples, 2))
    ensembles[0] = z0s

    for j in range(n_samples):
        sol = solve_ivp(_rhs, [0, T], z0s[j], method='RK45',
                        t_eval=t_eval, max_step=0.005,
                        rtol=1e-10, atol=1e-12)
        ensembles[:, j, :] = sol.y.T

    return t_eval, ensembles
