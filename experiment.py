#!/usr/bin/env python3
"""
experiment.py — Learn the Duffing Hamiltonian from noisy IC perturbations.

Pipeline:
  1. Perturb z₀ → z₀+ε, integrate forward by small dt, estimate the
     flow Jacobian DΦ_dt by least squares, extract ∇²H(z₀).
  2. Smooth the noisy Hessian field with NKN-kernel GPs (one per component).
  3. Double-integrate the smoothed Hessian to reconstruct H(q,p).
  4. Produce three visualisations:
       (a) Phase-space PDF evolution (ensemble spreading over time)
       (b) Learned NKN Hessian field vs ground truth
       (c) Reconstructed Hamiltonian contours vs true Hamiltonian
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from duffing import (hamiltonian, true_hessian, estimate_hessian_field,
                     evolve_ensemble, generate_velocity_data)
from kernels import RBFKernel, NKNKernel
from gp import fit, predict
from reconstruction import reconstruct

FIGDIR = 'figures/'

# ─── Configuration ───────────────────────────────────────────────────────────

Q_RANGE = (-1.6, 1.6)
P_RANGE = (-1.0, 1.0)

# Hessian estimation
N_EST_Q, N_EST_P = 10, 10         # grid for Jacobian estimation
DT_JAC = 0.1                     # small dt for linearisation
N_EPS = 40                       # perturbations per grid point
EPS_STD = 0.02                   # perturbation size

# GP smoothing grid (finer, for plotting)
N_FINE_Q, N_FINE_P = 35, 28

# NKN inducing grid
NKN_NQ, NKN_NP = 6, 8
MAX_ITER_NKN = 200

# PDF evolution
PDF_T = 2.0
PDF_SIGMA = 0.08
PDF_N_SAMPLES = 400
PDF_N_SAVE = 6


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    t_start = time.time()

    # ═════════════════════════════════════════════════════════════════════
    # STEP 1: Estimate ∇²H at grid points from IC perturbations
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1: Estimate Hessian field from IC perturbations")
    print("=" * 60)
    est_q = np.linspace(*Q_RANGE, N_EST_Q)
    est_p = np.linspace(*P_RANGE, N_EST_P)
    t0 = time.time()
    Z_est, Hqq_est, Hqp_est, Hpp_est, mean_vel = estimate_hessian_field(
        est_q, est_p, DT_JAC, N_EPS, EPS_STD, seed=42)
    print(f"  {len(Z_est)} grid points, {N_EPS} perturbations each")
    print(f"  Done in {time.time()-t0:.0f}s")

    # From mean velocity, recover ∇H at each estimation point:
    #   ż ≈ J∇H  →  ∇H = J^T ż = −J ż
    #   J = [[0,1],[-1,0]]  →  J^T = [[0,-1],[1,0]]
    # So: ∂H/∂q = −ż_p,  ∂H/∂p = ż_q
    grad_H_est = np.column_stack([-mean_vel[:, 1], mean_vel[:, 0]])

    # True values for comparison
    Hqq_true, Hqp_true, Hpp_true = true_hessian(Z_est[:, 0], Z_est[:, 1])
    err_qq = np.sqrt(np.mean((Hqq_est - Hqq_true)**2))
    err_pp = np.sqrt(np.mean((Hpp_est - Hpp_true)**2))
    print(f"  Raw estimation RMSE:  Hqq={err_qq:.4f}  Hpp={err_pp:.4f}")

    # ═════════════════════════════════════════════════════════════════════
    # STEP 2: Fit a SINGLE GP on H, observing its Hessian (NKN kernel)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2: Structured Hessian GP (single prior on H)")
    print("=" * 60)
    print("  One GP on H(z) with ∂⁴k-derived Gram matrix for ∇²H obs.")
    print("  This enforces integrability and symmetry by construction.\n")

    from hessian_gp import (stack_hessian_obs, hessian_fit, predict_H,
                            nkn_d4, nkn_d2_right, rbf_d4, rbf_d2_right)

    y_hess = stack_hessian_obs(Hqq_est, Hqp_est, Hpp_est)

    cq = np.linspace(*Q_RANGE, NKN_NQ)
    cp = np.linspace(*P_RANGE, NKN_NP)
    CQ, CP = np.meshgrid(cq, cp)
    centers = np.column_stack([CQ.ravel(), CP.ravel()])
    M = len(centers)

    le = np.zeros((M, 2))
    le[:, 0] = np.log(0.8); le[:, 1] = np.log(0.6)
    ll = np.zeros(M)

    nkn_H = NKNKernel(centers.copy(), ll.copy(), le.copy(), log_sn=np.log(0.3))
    print(f"  NKN: {M} inducing points, {nkn_H.n_params} params")
    print(f"  Gram matrix: {3*len(Z_est)} × {3*len(Z_est)}")
    print("  Fitting...")
    hessian_fit(nkn_H, Z_est, y_hess, nkn_d4, max_iter=MAX_ITER_NKN)
    print(f"    {nkn_H}")

    # ═════════════════════════════════════════════════════════════════════
    # STEP 3: Predict H directly on fine grid (no integration needed!)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3: Predict H on fine grid (direct posterior, no integration)")
    print("=" * 60)

    fine_q = np.linspace(*Q_RANGE, N_FINE_Q)
    fine_p = np.linspace(*P_RANGE, N_FINE_P)
    FQ, FP = np.meshgrid(fine_q, fine_p)
    Z_fine = np.column_stack([FQ.ravel(), FP.ravel()])
    fine_shape = (N_FINE_P, N_FINE_Q)

    mu_H, var_H = predict_H(nkn_H, Z_est, y_hess, Z_fine,
                             nkn_d4, nkn_d2_right)
    H_recon = mu_H.reshape(fine_shape)
    H_true = hamiltonian(FQ, FP)

    # ── Fix the affine ambiguity ─────────────────────────────────────────
    # Hessian obs pin curvature but leave a constant gradient offset
    # H → H + a·z + c.  We fix both using the velocity-derived ∇H.
    #
    # 1. Compute GP posterior's gradient at estimation points (numerical)
    # 2. Compare to the true ∇H from mean velocity
    # 3. The constant offset a = mean(∇H_true − ∇H_GP) over all points

    # Also predict H at the estimation points for gradient comparison
    mu_H_est, _ = predict_H(nkn_H, Z_est, y_hess, Z_est,
                             nkn_d4, nkn_d2_right)

    # Numerical gradient of GP posterior at estimation points
    # Use finite differences with a small step
    h_fd = 0.01
    grad_H_gp = np.zeros((len(Z_est), 2))
    for dim in range(2):
        Z_plus = Z_est.copy(); Z_plus[:, dim] += h_fd
        Z_minus = Z_est.copy(); Z_minus[:, dim] -= h_fd
        mu_plus, _ = predict_H(nkn_H, Z_est, y_hess, Z_plus,
                                nkn_d4, nkn_d2_right)
        mu_minus, _ = predict_H(nkn_H, Z_est, y_hess, Z_minus,
                                 nkn_d4, nkn_d2_right)
        grad_H_gp[:, dim] = (mu_plus - mu_minus) / (2 * h_fd)

    # Constant gradient offset: average discrepancy
    grad_offset = np.mean(grad_H_est - grad_H_gp, axis=0)
    print(f"  Gradient offset (a):  dH/dq = {grad_offset[0]:.4f}, "
          f"dH/dp = {grad_offset[1]:.4f}")

    # Apply linear correction to H on fine grid
    H_recon += grad_offset[0] * FQ + grad_offset[1] * FP

    # Fix additive constant by aligning at saddle
    iq0 = np.argmin(np.abs(fine_q))
    ip0 = np.argmin(np.abs(fine_p))
    offset = H_true[ip0, iq0] - H_recon[ip0, iq0]
    H_recon += offset
    print(f"  Additive constant offset: {offset:.6f}")

    recon_err = np.sqrt(np.mean((H_recon - H_true)**2))
    print(f"  Reconstruction RMSE: {recon_err:.6f}")

    # Also get true Hessian for plotting
    Hqq_t, Hqp_t, Hpp_t = true_hessian(FQ, FP)

    # ═════════════════════════════════════════════════════════════════════
    # STEP 4: Generate PDF evolution data
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4: PDF evolution (ensemble propagation)")
    print("=" * 60)

    # Three starting points: near saddle, left well, outside separatrix
    z0_list = [
        (np.array([0.1, 0.2]),  'Near saddle'),
        (np.array([-1.0, 0.0]), 'Left well'),
        (np.array([0.0, 0.9]),  'Outside separatrix'),
    ]

    ensemble_data = []
    for z0, label in z0_list:
        print(f"  Evolving ensemble from {label}  z₀={z0}...")
        t0 = time.time()
        times, ens = evolve_ensemble(z0, PDF_SIGMA, PDF_N_SAMPLES,
                                     PDF_T, PDF_N_SAVE, seed=0)
        print(f"    {PDF_N_SAMPLES} samples, {PDF_N_SAVE} snapshots ({time.time()-t0:.0f}s)")
        ensemble_data.append((times, ens, label))

    # ═════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 5: Generating figures")
    print("=" * 60)

    # ─── (a) Phase-space PDF evolution ───────────────────────────────────
    n_starts = len(z0_list)
    n_snaps = PDF_N_SAVE
    fig, axes = plt.subplots(n_starts, n_snaps, figsize=(4*n_snaps, 4*n_starts))

    for row, (times, ens, label) in enumerate(ensemble_data):
        for col in range(n_snaps):
            ax = axes[row, col]
            # Energy contours
            qq = np.linspace(-2.2, 2.2, 200)
            pp = np.linspace(-1.5, 1.5, 200)
            QQ, PP = np.meshgrid(qq, pp)
            ax.contour(QQ, PP, hamiltonian(QQ, PP), levels=[0],
                       colors='red', linewidths=1, linestyles='--')
            ax.contour(QQ, PP, hamiltonian(QQ, PP),
                       levels=np.linspace(-0.25, 1.0, 12),
                       colors='gray', linewidths=0.3, alpha=0.3)
            # Ensemble scatter
            ax.scatter(ens[col, :, 0], ens[col, :, 1],
                       s=1, alpha=0.3, c='steelblue')
            ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f't = {times[col]:.2f}', fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
            ax.tick_params(labelsize=7)

    fig.suptitle('Phase-Space PDF Evolution (IC Noise Propagation)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/pdf_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → pdf_evolution.png")

    # ─── (b) Learned Hessian field vs truth ──────────────────────────────
    # Since we now have a single GP on H, we show H directly (no separate
    # Hessian component plots needed — the Hessian is implicitly correct
    # by construction). But for comparison, we can numerically differentiate
    # the predicted H to show the implied Hessian.

    # Numerical Hessian of the reconstructed H
    dq = fine_q[1] - fine_q[0]
    dp = fine_p[1] - fine_p[0]
    Hqq_recon = np.gradient(np.gradient(H_recon, dq, axis=1), dq, axis=1)
    Hqp_recon = np.gradient(np.gradient(H_recon, dp, axis=0), dq, axis=1)
    Hpp_recon = np.gradient(np.gradient(H_recon, dp, axis=0), dp, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles_top = ['Implied H_qq (from GP on H)', 'Implied H_qp', 'Implied H_pp']
    titles_bot = ['True H_qq', 'True H_qp', 'True H_pp']
    est_fields = [Hqq_recon, Hqp_recon, Hpp_recon]
    true_fields = [Hqq_t, Hqp_t, Hpp_t]

    for col, (est, tru, t1, t2) in enumerate(zip(
            est_fields, true_fields, titles_top, titles_bot)):
        vmin = min(est.min(), tru.min())
        vmax = max(est.max(), tru.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        ax = axes[0, col]
        pcm = ax.pcolormesh(FQ, FP, est, shading='auto', cmap='RdBu_r', norm=norm)
        fig.colorbar(pcm, ax=ax, shrink=0.8)
        ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
                   colors='black', linewidths=1, linestyles='--')
        ax.set_title(f'{t1} (NKN GP)', fontsize=12)
        ax.set_xlabel('q'); ax.set_ylabel('p')

        ax = axes[1, col]
        pcm = ax.pcolormesh(FQ, FP, tru, shading='auto', cmap='RdBu_r', norm=norm)
        fig.colorbar(pcm, ax=ax, shrink=0.8)
        ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
                   colors='black', linewidths=1, linestyles='--')
        ax.set_title(f'{t2} (analytic)', fontsize=12)
        ax.set_xlabel('q'); ax.set_ylabel('p')

    # Overlay raw estimates on top-left panel
    shape_est = (N_EST_P, N_EST_Q)
    axes[0, 0].scatter(Z_est[:, 0], Z_est[:, 1], c=Hqq_est,
                       s=15, edgecolors='k', linewidths=0.3,
                       cmap='RdBu_r', norm=Normalize(
                           vmin=est_fields[0].min(), vmax=est_fields[0].max()),
                       zorder=5)

    fig.suptitle('Learned Hessian Field: NKN GP vs Analytic Truth',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/hessian_field.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → hessian_field.png")

    # ─── (c) Reconstructed Hamiltonian ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    levels = np.linspace(-0.3, 1.5, 20)

    ax = axes[0]
    cs = ax.contourf(FQ, FP, H_recon, levels=levels, cmap='viridis')
    ax.contour(FQ, FP, H_recon, levels=[0], colors='red', linewidths=2)
    fig.colorbar(cs, ax=ax, shrink=0.85)
    ax.set_title('Reconstructed H (from NKN)', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    ax = axes[1]
    cs = ax.contourf(FQ, FP, H_true, levels=levels, cmap='viridis')
    ax.contour(FQ, FP, H_true, levels=[0], colors='red', linewidths=2)
    fig.colorbar(cs, ax=ax, shrink=0.85)
    ax.set_title('True H', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    ax = axes[2]
    err = H_recon - H_true
    elim = max(abs(err.min()), abs(err.max()))
    cs = ax.contourf(FQ, FP, err, levels=np.linspace(-elim, elim, 20),
                     cmap='RdBu_r')
    fig.colorbar(cs, ax=ax, shrink=0.85, label='H_recon − H_true')
    ax.contour(FQ, FP, H_true, levels=[0], colors='black',
               linewidths=1, linestyles='--')
    ax.set_title(f'Error  (RMSE = {recon_err:.4f})', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    fig.suptitle('Hamiltonian Reconstruction from IC-Noise Hessian Estimation',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/hamiltonian_reconstruction.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → hamiltonian_reconstruction.png")

    # ─── (c2) Posterior covariance of H ──────────────────────────────────
    #
    # The GP posterior variance Var[H(z)] shows where the model is
    # uncertain about the Hamiltonian. This should be large far from
    # training points and near the separatrix (where the Hessian
    # estimates are noisier).
    #
    sigma_H = np.sqrt(var_H).reshape(fine_shape)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Posterior std dev σ[H]
    ax = axes[0]
    pcm = ax.pcolormesh(FQ, FP, sigma_H, shading='auto', cmap='magma')
    ax.contour(FQ, FP, H_true, levels=[0], colors='cyan',
               linewidths=1.5, linestyles='--')
    ax.scatter(Z_est[:, 0], Z_est[:, 1], s=8, c='white',
               edgecolors='none', alpha=0.6, zorder=5)
    fig.colorbar(pcm, ax=ax, shrink=0.85, label='σ[H]')
    ax.set_title('Posterior Std Dev  σ[H(q,p)]', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    # Panel 2: Reconstructed H with ±2σ contours
    ax = axes[1]
    cs = ax.contourf(FQ, FP, H_recon, levels=np.linspace(-0.3, 1.5, 20),
                     cmap='viridis', alpha=0.8)
    fig.colorbar(cs, ax=ax, shrink=0.85)
    # ±2σ bands around the separatrix (H=0 level set)
    ax.contour(FQ, FP, H_recon, levels=[0], colors='red', linewidths=2)
    ax.contour(FQ, FP, H_recon - 2*sigma_H, levels=[0], colors='red',
               linewidths=1, linestyles='--', alpha=0.7)
    ax.contour(FQ, FP, H_recon + 2*sigma_H, levels=[0], colors='red',
               linewidths=1, linestyles='--', alpha=0.7)
    ax.set_title('Reconstructed H with ±2σ Separatrix Band', fontsize=12)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    # Panel 3: Relative uncertainty σ[H] / |H_true| (where |H| > threshold)
    ax = axes[2]
    H_abs = np.abs(H_true)
    rel_unc = np.where(H_abs > 0.02, sigma_H / H_abs, np.nan)
    pcm = ax.pcolormesh(FQ, FP, rel_unc, shading='auto', cmap='inferno',
                        vmin=0, vmax=min(np.nanmax(rel_unc), 2.0))
    ax.contour(FQ, FP, H_true, levels=[0], colors='cyan',
               linewidths=1.5, linestyles='--')
    fig.colorbar(pcm, ax=ax, shrink=0.85, label='σ[H] / |H|')
    ax.set_title('Relative Uncertainty  σ[H]/|H|', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    fig.suptitle('Posterior Uncertainty in the Reconstructed Hamiltonian',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/posterior_covariance.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → posterior_covariance.png")

    # ─── (c3) Covariance cross-sections ──────────────────────────────────
    #
    # Show Cov[H(z_ref), H(z)] for a few reference points — how the
    # posterior uncertainty at one point correlates with other locations.
    #
    from hessian_gp import build_H_cross, nkn_d2_right as d2_fn

    ref_points = [
        (np.array([0.0, 0.0]),  'Saddle (0, 0)'),
        (np.array([-1.0, 0.0]), 'Left well (−1, 0)'),
        (np.array([0.5, 0.5]),  'Near separatrix (0.5, 0.5)'),
    ]

    # Full posterior covariance is expensive, so compute cross-sections:
    # Cov_post[H(z_ref), H(z_*)] = k(z_ref, z_*) - k_ref^T (K+σ²I)^{-1} k_*
    # where k_ref = cross(z_ref, Z_train), k_* = cross(z_*, Z_train)
    from scipy.linalg import cho_factor, cho_solve
    N_est = len(Z_est)
    K_gram = np.zeros((3*N_est, 3*N_est))
    # Rebuild gram (fast with vectorised NKN)
    from hessian_gp import build_hessian_gram, nkn_d4
    K_gram = build_hessian_gram(nkn_H, Z_est, nkn_d4)
    K_gram += (nkn_H.sigma_n**2 + 1e-6) * np.eye(3*N_est)
    L_gram, lower_gram = cho_factor(K_gram, lower=True)

    fig, axes = plt.subplots(1, len(ref_points), figsize=(7*len(ref_points), 5.5))

    for ax, (z_ref, label) in zip(axes, ref_points):
        # k(z_ref, z_*) for all z_* on fine grid (prior cross-cov)
        k_ref_star = np.zeros(len(Z_fine))
        for m in range(len(Z_fine)):
            k_ref_star[m] = nkn_H.gram(z_ref.reshape(1, -1),
                                        Z_fine[m].reshape(1, -1))[0, 0]

        # k_ref = cross(z_ref, Z_train) — Cov[H(z_ref), Hessian obs]
        k_ref_train = build_H_cross(nkn_H, z_ref.reshape(1, -1),
                                     Z_est, d2_fn)  # (1, 3N)

        # k_star = cross(Z_fine, Z_train) — already needed but expensive
        # Approximate: compute Cov[H(z_ref), H(z_*)] via the formula
        # = k(z_ref, z_*) - k_ref_train @ K^{-1} @ k_star_train^T
        # where k_star_train for each z_* is (1, 3N)
        cov_section = np.zeros(len(Z_fine))
        K_inv_kref = cho_solve((L_gram, lower_gram), k_ref_train.T)  # (3N, 1)

        for m in range(len(Z_fine)):
            k_star_train = build_H_cross(nkn_H, Z_fine[m].reshape(1, -1),
                                          Z_est, d2_fn)  # (1, 3N)
            cov_section[m] = k_ref_star[m] - (k_star_train @ K_inv_kref)[0, 0]

        C = cov_section.reshape(fine_shape)
        clim = max(abs(C.min()), abs(C.max()))
        pcm = ax.pcolormesh(FQ, FP, C, shading='auto', cmap='RdBu_r',
                            vmin=-clim, vmax=clim)
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.contour(FQ, FP, H_true, levels=[0], colors='black',
                   linewidths=1, linestyles='--')
        ax.plot(z_ref[0], z_ref[1], 'w*', ms=15, mew=1.5, zorder=10)
        ax.set_xlabel('q'); ax.set_ylabel('p')
        ax.set_title(f'Cov[H(★), H(q,p)]\n★ = {label}', fontsize=11)

    fig.suptitle('Posterior Cross-Covariance of H: How Uncertainty Correlates Across Phase Space',
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/posterior_cross_covariance.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → posterior_cross_covariance.png")

    # ─── (d) NKN kernel structure ────────────────────────────────────────
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    k = nkn_H
    ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
               colors='red', linewidths=1, linestyles='--')
    ax.contour(FQ, FP, hamiltonian(FQ, FP),
               levels=np.linspace(-0.25, 1.0, 12),
               colors='gray', linewidths=0.3, alpha=0.3)
    amps = np.exp(k.log_lam)
    ells = np.exp(k.log_ell)
    for i in range(k.M):
        alpha_i = max(amps[i] / max(amps.max(), 1e-12), 0.1)
        e = Ellipse(xy=k.centers[i],
                    width=2*ells[i, 0], height=2*ells[i, 1],
                    fill=False, edgecolor='dodgerblue',
                    linewidth=1.5, alpha=alpha_i)
        ax.add_patch(e)
    sc = ax.scatter(k.centers[:, 0], k.centers[:, 1],
                    c=amps, s=60, cmap='plasma', edgecolors='k', zorder=5)
    plt.colorbar(sc, ax=ax, label='Amplitude λ_i')
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.5, 1.5)
    ax.set_title('NKN Kernel on H: Inducing Points & Bandwidths', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    fig.suptitle('NKN Kernel Structure (Inducing Points & Bandwidths)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/nkn_structure.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → nkn_structure.png")

    # ─── (e) NKN prior variance k(z,z) as heteroscedastic diagnostic ───
    #
    # The NKN's defining advantage: k_NKN(z,z) = Σ_i λ_i φ_i(z-p_i)²
    # is POSITION-DEPENDENT, while k_RBF(z,z) = σ_f² is constant.
    #
    # After fitting the kernel to noisy velocity data, the NKN's local
    # prior variance k(z,z) reflects where the kernel allocates signal
    # power — which, for heteroscedastic data, tracks where the noise is
    # large. This is the NKN's LEARNED spatial structure, directly from
    # the kernel parameters, not from a separate regression.
    #
    print("\n  ── Fitting velocity kernels (RBF vs NKN) ──")

    N_VEL_REPS = 5
    Z_vel, qdot_obs, pdot_obs = generate_velocity_data(
        est_q, est_p, n_eps=N_VEL_REPS, eps_std=EPS_STD, seed=99)
    N_grid = N_EST_Q * N_EST_P
    print(f"  {len(Z_vel)} velocity observations ({N_grid} × {N_VEL_REPS} reps)")

    vel_results = {}  # {(comp, kernel_name): variance_at_fine_grid}

    for comp_name, y_obs in [('qdot', qdot_obs), ('pdot', pdot_obs)]:
        # ── RBF ──
        rbf = RBFKernel(log_sf=0.0, log_ell=np.log(0.5), log_sn=np.log(0.05))
        print(f"  Fitting RBF for {comp_name}...")
        fit(rbf, Z_vel, y_obs, max_iter=200)
        # RBF prior variance: k(z,z) = σ_f² — constant everywhere
        rbf_prior_var = rbf.sigma_f**2 * np.ones(len(Z_fine))
        vel_results[(comp_name, 'RBF')] = rbf_prior_var
        print(f"    {rbf}  →  k(z,z) = {rbf.sigma_f**2:.6f} (constant)")

        # ── NKN ──
        vc = np.linspace(*Q_RANGE, 4)
        vp = np.linspace(*P_RANGE, 3)
        VCQ, VCP = np.meshgrid(vc, vp)
        vel_centers = np.column_stack([VCQ.ravel(), VCP.ravel()])
        Mv = len(vel_centers)
        vle = np.zeros((Mv, 2)); vle[:, 0] = np.log(0.5); vle[:, 1] = np.log(0.4)
        vll = np.zeros(Mv)
        nkn_vel = NKNKernel(vel_centers.copy(), vll.copy(), vle.copy(),
                            log_sn=np.log(0.05))
        print(f"  Fitting NKN for {comp_name}...")
        fit(nkn_vel, Z_vel, y_obs, max_iter=300)
        # NKN prior variance: k(z,z) = Σ_i λ_i φ_i(z-p_i)² — position-dependent
        nkn_prior_var = np.diag(nkn_vel.gram(Z_fine, Z_fine))
        vel_results[(comp_name, 'NKN')] = nkn_prior_var
        vel_results[(comp_name, 'NKN_kernel')] = nkn_vel
        vel_results[(comp_name, 'RBF_kernel')] = rbf
        vel_results[(comp_name, 'y_obs')] = y_obs
        print(f"    {nkn_vel}  →  k(z,z) ∈ [{nkn_prior_var.min():.4f}, {nkn_prior_var.max():.4f}]")

    # ── Compute posterior variances via predict() ────────────────────────
    #
    # The POSTERIOR variance is k(z*,z*) - k*^T (K+σ²I)^{-1} k*
    # Unlike the prior k(z,z), this is reduced near training points.
    # For the NKN it's position-dependent in BOTH prior and data terms.
    #
    print("\n  Computing posterior variances...")
    post_results = {}
    for comp_name in ['qdot', 'pdot']:
        y_obs = vel_results[(comp_name, 'y_obs')]
        for kname in ['RBF', 'NKN']:
            kernel = vel_results[(comp_name, f'{kname}_kernel')]
            _, post_var = predict(kernel, Z_vel, y_obs, Z_fine)
            post_results[(comp_name, kname)] = post_var
            print(f"    {kname} posterior Var[{comp_name}]: "
                  f"[{post_var.min():.6f}, {post_var.max():.6f}]")

    # ── Plot: 2×2 grid (rows = qdot/pdot, cols = RBF/NKN) ───────────────
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for row, comp in enumerate(['qdot', 'pdot']):
        for col, kname in enumerate(['RBF', 'NKN']):
            ax = axes[row, col]
            var = vel_results[(comp, kname)]
            V = var.reshape(fine_shape)
            pcm = ax.pcolormesh(FQ, FP, V, shading='auto', cmap='inferno',
                                norm=LogNorm(vmin=vmin))
            fig.colorbar(pcm, ax=ax, shrink=0.85)
            ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
                       colors='cyan', linewidths=1.5, linestyles='--')
            ax.set_xlabel('q'); ax.set_ylabel('p')

            comp_label = r'$\dot{q}$' if comp == 'qdot' else r'$\dot{p}$'
            ax.set_title(f'{kname} — k(z,z) for {comp_label}', fontsize=12)

    fig.suptitle(
        r'Learned Kernel Prior Variance k(z,z): RBF (constant) vs NKN (position-dependent)',
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/velocity_variance.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → velocity_variance.png")

    # ── Plot: side-by-side with true heteroscedastic noise field ─────────
    # True first-order noise variance for pdot:
    #   Var[pdot(z₀+ε)] ≈ ε_std² (1 − 3q₀²)²   (from Taylor expansion)
    true_pdot_var = EPS_STD**2 * (1 - 3*FQ**2)**2

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    var_rbf_p = vel_results[('pdot', 'RBF')]
    var_nkn_p = vel_results[('pdot', 'NKN')]

    all_v = np.concatenate([var_rbf_p, var_nkn_p, true_pdot_var.ravel()])
    vmin = max(all_v[all_v > 0].min(), 1e-10)
    vmax = all_v.max()

    for ax, v, title in zip(axes,
                            [var_rbf_p.reshape(fine_shape),
                             var_nkn_p.reshape(fine_shape),
                             true_pdot_var],
                            [r'RBF — k(z,z) = $\sigma_f^2$ (constant)',
                             r'NKN — k(z,z) = $\Sigma_i \lambda_i \varphi_i^2$ (learned)',
                             r'True noise: $\sigma^2(1-3q^2)^2$']):
        V = np.clip(v, vmin, None)
        pcm = ax.pcolormesh(FQ, FP, V, shading='auto', cmap='inferno',
                            norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
                   colors='cyan', linewidths=1.5, linestyles='--')
        ax.set_xlabel('q'); ax.set_ylabel('p')
        ax.set_title(title, fontsize=13)

    fig.suptitle(
        r'$\dot{p}$ Kernel Prior Variance: RBF (constant) vs NKN (learned) vs Truth',
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/pdot_variance_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → pdot_variance_comparison.png")

    # ── Posterior variance 2×2: qdot/pdot × RBF/NKN ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for row, comp in enumerate(['qdot', 'pdot']):
        for col, kname in enumerate(['RBF', 'NKN']):
            ax = axes[row, col]
            pv = post_results[(comp, kname)]
            V = pv.reshape(fine_shape)
            vmin_pv = max(V[V > 0].min(), 1e-12)
            pcm = ax.pcolormesh(FQ, FP, V, shading='auto', cmap='inferno',
                                norm=LogNorm(vmin=vmin_pv))
            fig.colorbar(pcm, ax=ax, shrink=0.85)
            ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
                       colors='cyan', linewidths=1.5, linestyles='--')
            ax.scatter(Z_vel[::N_VEL_REPS, 0], Z_vel[::N_VEL_REPS, 1],
                       s=5, c='white', alpha=0.4, zorder=5)
            ax.set_xlabel('q'); ax.set_ylabel('p')
            comp_label = r'$\dot{q}$' if comp == 'qdot' else r'$\dot{p}$'
            ax.set_title(f'{kname} — Posterior Var[{comp_label}]', fontsize=12)

    fig.suptitle(
        r'Posterior Predictive Variance: $k(z,z) - k_*^T(K+\sigma^2I)^{-1}k_*$',
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/posterior_velocity_variance.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → posterior_velocity_variance.png")

    # ── Posterior pdot comparison: RBF vs NKN vs truth ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    post_rbf_p = post_results[('pdot', 'RBF')]
    post_nkn_p = post_results[('pdot', 'NKN')]

    all_pv = np.concatenate([post_rbf_p, post_nkn_p, true_pdot_var.ravel()])
    vmin_post = max(all_pv[all_pv > 0].min(), 1e-12)
    vmax_post = all_pv.max()

    for ax, v, title in zip(axes,
                            [post_rbf_p.reshape(fine_shape),
                             post_nkn_p.reshape(fine_shape),
                             true_pdot_var],
                            [r'RBF — Posterior Var[$\dot{p}$]',
                             r'NKN — Posterior Var[$\dot{p}$]',
                             r'True noise: $\sigma^2(1-3q^2)^2$']):
        V = np.clip(v, vmin_post, None)
        pcm = ax.pcolormesh(FQ, FP, V, shading='auto', cmap='inferno',
                            norm=LogNorm(vmin=vmin_post, vmax=vmax_post))
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.contour(FQ, FP, hamiltonian(FQ, FP), levels=[0],
                   colors='cyan', linewidths=1.5, linestyles='--')
        ax.set_xlabel('q'); ax.set_ylabel('p')
        ax.set_title(title, fontsize=12)

    fig.suptitle(
        r'$\dot{p}$ Posterior Variance: RBF vs NKN vs Analytic Truth',
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/pdot_posterior_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  → pdot_posterior_comparison.png")

    # ─── (g) Kernel slices: k(q,q') and k(p,p') ─────────────────────────
    #
    # Show the kernel function itself to see stationarity vs nonstationarity.
    # For k(q,q'): fix p = p' = 0, sweep q and q'.
    # For k(p,p'): fix q = q' = 0, sweep p and p'.
    # RBF: k depends only on |z−z'| → constant along diagonals.
    # NKN: k varies with absolute position → asymmetric, position-dependent.
    #
    print("\n  ── Plotting kernel slices ──")

    # Use the pdot kernels (where nonstationarity matters most)
    # Re-fit quickly to get clean kernel objects
    rbf_for_slice = RBFKernel(log_sf=0.0, log_ell=np.log(0.5), log_sn=np.log(0.05))
    fit(rbf_for_slice, Z_vel, pdot_obs, max_iter=200, verbose=False)

    nkn_for_slice = NKNKernel(vel_centers.copy(), vll.copy(), vle.copy(),
                               log_sn=np.log(0.05))
    fit(nkn_for_slice, Z_vel, pdot_obs, max_iter=300, verbose=False)

    slice_pts = np.linspace(*Q_RANGE, 60)
    p_slice_pts = np.linspace(*P_RANGE, 60)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # ── Row 0: k(q, q') at fixed p = p' = 0 ──
    for col, (kernel, kname) in enumerate([(rbf_for_slice, 'RBF'),
                                            (nkn_for_slice, 'NKN')]):
        ax = axes[0, col]
        n = len(slice_pts)
        K_slice = np.zeros((n, n))
        for i in range(n):
            z_i = np.array([[slice_pts[i], 0.0]])
            for j in range(n):
                z_j = np.array([[slice_pts[j], 0.0]])
                K_slice[i, j] = kernel.gram(z_i, z_j)[0, 0]

        pcm = ax.pcolormesh(slice_pts, slice_pts, K_slice, shading='auto',
                            cmap='viridis')
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.set_xlabel("q"); ax.set_ylabel("q'")
        ax.set_title(f"{kname} — k(q, q') at p = p' = 0", fontsize=12)
        ax.set_aspect('equal')
        # Mark the elliptic-hyperbolic transition
        for qval in [-1/np.sqrt(3), 1/np.sqrt(3)]:
            ax.axvline(qval, color='red', ls='--', lw=0.8, alpha=0.5)
            ax.axhline(qval, color='red', ls='--', lw=0.8, alpha=0.5)

    # ── Row 1: k(p, p') at fixed q = q' for two different q values ──
    q_vals_for_slice = [0.0, 1.0]  # saddle vs well
    for col, q_fix in enumerate(q_vals_for_slice):
        ax = axes[1, col]
        n = len(p_slice_pts)

        # NKN kernel slice
        K_nkn = np.zeros((n, n))
        K_rbf = np.zeros((n, n))
        for i in range(n):
            z_i = np.array([[q_fix, p_slice_pts[i]]])
            for j in range(n):
                z_j = np.array([[q_fix, p_slice_pts[j]]])
                K_nkn[i, j] = nkn_for_slice.gram(z_i, z_j)[0, 0]
                K_rbf[i, j] = rbf_for_slice.gram(z_i, z_j)[0, 0]

        # Plot NKN with RBF contours overlaid
        pcm = ax.pcolormesh(p_slice_pts, p_slice_pts, K_nkn, shading='auto',
                            cmap='viridis')
        ax.contour(p_slice_pts, p_slice_pts, K_rbf, levels=5,
                   colors='white', linewidths=0.8, linestyles='--', alpha=0.6)
        fig.colorbar(pcm, ax=ax, shrink=0.85)
        ax.set_xlabel("p"); ax.set_ylabel("p'")
        location = "saddle (q=0)" if q_fix == 0.0 else "well (q=1)"
        ax.set_title(f"NKN k(p, p') at q = q' = {q_fix} ({location})",
                     fontsize=11)
        ax.set_aspect('equal')

    fig.suptitle(
        "Kernel Structure: RBF (stationary) vs NKN (learned, nonstationary)",
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kernel_slices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → kernel_slices.png")

    # ── 1D diagonal slices: k(q, q+δ) as a function of δ at different q ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    deltas = np.linspace(-1.5, 1.5, 100)
    q_centers = [-1.0, -0.4, 0.0, 0.4, 1.0]
    colors_q = plt.cm.coolwarm(np.linspace(0, 1, len(q_centers)))

    for ax, (kernel, kname) in zip(axes, [(rbf_for_slice, 'RBF'),
                                           (nkn_for_slice, 'NKN')]):
        for qi, c in zip(q_centers, colors_q):
            k_vals = []
            for d in deltas:
                z1 = np.array([[qi, 0.0]])
                z2 = np.array([[qi + d, 0.0]])
                k_vals.append(kernel.gram(z1, z2)[0, 0])
            ax.plot(deltas, k_vals, color=c, lw=2,
                    label=f'q₀ = {qi:.1f}')

        ax.set_xlabel('δ (displacement from q₀)')
        ax.set_ylabel("k(q₀, q₀ + δ) at p = 0")
        ax.set_title(f'{kname} Kernel', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "1D Kernel Profiles: k(q₀, q₀+δ) at Different Base Points",
        fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/kernel_profiles.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → kernel_profiles.png")

    # Save kernel parameters for interactive visualization
    import json
    kernel_data = {
        'rbf': {
            'sigma_f': float(rbf_for_slice.sigma_f),
            'ell': float(rbf_for_slice.ell),
        },
        'nkn': {
            'M': int(nkn_for_slice.M),
            'log_lam': nkn_for_slice.log_lam.tolist(),
            'log_ell': nkn_for_slice.log_ell.tolist(),
            'centers': nkn_for_slice.centers.tolist(),
        },
        'q_range': list(Q_RANGE),
        'p_range': list(P_RANGE),
    }
    with open(f'{FIGDIR}/kernel_params.json', 'w') as f:
        json.dump(kernel_data, f)
    print("  → kernel_params.json")

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Hessian estimation points:  {len(Z_est)}")
    print(f"  Raw Hqq RMSE:              {err_qq:.4f}")
    print(f"  Raw Hpp RMSE:              {err_pp:.4f}")
    print(f"  H reconstruction RMSE:     {recon_err:.6f}")
    print(f"  Total time:                {time.time()-t_start:.0f}s")
    print(f"  NKN (H): {nkn_H}")
    print(f"\n  Figures in {FIGDIR}")


if __name__ == '__main__':
    print("=" * 60)
    print("  Duffing Hamiltonian Reconstruction from IC Noise")
    print("=" * 60 + "\n")
    main()