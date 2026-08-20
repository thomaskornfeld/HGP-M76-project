#!/usr/bin/env python3
"""
experiment.py — Hamiltonian reconstruction via joint gradient+Hessian GP.

Follows the paper: condition on (∂qH, ∂pH, ∂q²H, ∂q∂pH, ∂p²H) at each
estimation point, with two noise levels (σ_grad, σ_hess). The gradient
observations resolve the affine ambiguity automatically — no post-hoc
correction needed.
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Ellipse

from duffing import (hamiltonian, true_hessian, estimate_hessian_field,
                     evolve_ensemble, generate_velocity_data)
from kernels import RBFKernel, NKNKernel
from gp import fit, predict
from hamiltonian_gp import (stack_observations, build_noise_matrix,
                            joint_fit_empirical, predict_H_empirical,
                            joint_fit, predict_H,
                            build_gram, noise_diag)

FIGDIR = 'figures/'

# ─── Config ──────────────────────────────────────────────────────────────────

Q_RANGE = (-1.6, 1.6)
P_RANGE = (-1.0, 1.0)

N_EST_Q, N_EST_P = 8, 6
DT_JAC   = 0.1
N_EPS    = 40
EPS_STD  = 0.02

N_FINE_Q, N_FINE_P = 35, 28
NKN_NQ, NKN_NP = 4, 3
MAX_ITER = 300

PDF_T = 2.0; PDF_SIGMA = 0.08; PDF_N_SAMPLES = 400; PDF_N_SAVE = 6


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    t_start = time.time()

    # ═════════════════════════════════════════════════════════════════════
    # STEP 1: Estimate gradient + Hessian from IC perturbations
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("STEP 1: Estimate ∇H and ∇²H from IC perturbations")
    print("=" * 60)
    est_q = np.linspace(*Q_RANGE, N_EST_Q)
    est_p = np.linspace(*P_RANGE, N_EST_P)
    t0 = time.time()
    Z_est, Hqq_est, Hqp_est, Hpp_est, mean_vel, noise_hess, noise_grad = \
        estimate_hessian_field(est_q, est_p, DT_JAC, N_EPS, EPS_STD, seed=42)
    N_est = len(Z_est)
    print(f"  {N_est} points, {N_EPS} perturbations each ({time.time()-t0:.0f}s)")

    # Gradient from mean drift: ∇H = J^T ż = (−ṗ, q̇)
    grad_H_est = np.column_stack([-mean_vel[:, 1], mean_vel[:, 0]])

    # True values
    Hqq_true, _, _ = true_hessian(Z_est[:, 0], Z_est[:, 1])
    grad_H_true = np.column_stack([Z_est[:, 0]**3 - Z_est[:, 0], Z_est[:, 1]])

    print(f"  Gradient RMSE:  {np.sqrt(np.mean((grad_H_est - grad_H_true)**2)):.6f}")
    print(f"  Hqq RMSE:      {np.sqrt(np.mean((Hqq_est - Hqq_true)**2)):.6f}")

    # ═════════════════════════════════════════════════════════════════════
    # STEP 2: Fit BOTH RBF and NKN via the joint 5-component GP
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2: Joint GP  (5 obs per point: ∇H + ∇²H)")
    print("=" * 60)

    y = stack_observations(grad_H_est, Hqq_est, Hqp_est, Hpp_est)
    noise_mat = build_noise_matrix(noise_grad, noise_hess)
    print(f"  Observation vector: {len(y)} entries ({N_est} x 5)")
    print(f"  Noise: empirical per-point 5x5 blocks (no fitted noise params)")

    cq = np.linspace(*Q_RANGE, NKN_NQ)
    cp = np.linspace(*P_RANGE, NKN_NP)
    CQ, CP = np.meshgrid(cq, cp)
    centers = np.column_stack([CQ.ravel(), CP.ravel()])

    # ── Fit NKN (kernel params only — noise is empirical, fixed) ─────────
    print("\n  ── NKN kernel (empirical noise) ──")
    nkn = NKNKernel(centers.copy(), log_sn=np.log(0.3))
    nkn.chol[:, 0] = np.log(0.8)
    nkn.chol[:, 1] = 0.0
    nkn.chol[:, 2] = np.log(0.6)
    print(f"  {nkn.M} inducing points, {nkn.n_params} kernel params (noise fixed)")
    res_nkn = joint_fit_empirical(nkn, Z_est, y, noise_mat, max_iter=MAX_ITER)
    print(f"    {nkn}")

    # ── Fit RBF (kernel params only — same empirical noise) ──────────────
    print("\n  ── RBF kernel (empirical noise, sigma_f bounded) ──")
    rbf = RBFKernel(log_sf=0.0, log_ell=np.log(0.5), log_sn=np.log(0.1))
    rbf_bounds = [(None, 2.0), (None, None), (None, None)]
    res_rbf = joint_fit_empirical(rbf, Z_est, y, noise_mat,
                                   max_iter=MAX_ITER, bounds=rbf_bounds)
    print(f"    {rbf}")


    # ═════════════════════════════════════════════════════════════════════
    # STEP 3: Predict H with both kernels
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3: Predict H (RBF vs NKN)")
    print("=" * 60)

    fine_q = np.linspace(*Q_RANGE, N_FINE_Q)
    fine_p = np.linspace(*P_RANGE, N_FINE_P)
    FQ, FP = np.meshgrid(fine_q, fine_p)
    Z_fine = np.column_stack([FQ.ravel(), FP.ravel()])
    fine_shape = (N_FINE_P, N_FINE_Q)
    H_true = hamiltonian(FQ, FP)
    Hqq_t, Hqp_t, Hpp_t = true_hessian(FQ, FP)

    iq0 = np.argmin(np.abs(fine_q))
    ip0 = np.argmin(np.abs(fine_p))

    results = {}  # {name: (H_recon, var_H, prior_var, rmse)}

    for name, kern in [('NKN', nkn), ('RBF', rbf)]:
        print(f"\n  Predicting with {name}...")
        mu_H, var_H = predict_H_empirical(kern, Z_est, y, Z_fine, noise_mat)
        H_rec = mu_H.reshape(fine_shape)
        # Fix additive constant
        offset = H_true[ip0, iq0] - H_rec[ip0, iq0]
        H_rec += offset
        rmse = np.sqrt(np.mean((H_rec - H_true)**2))
        # Prior variance k(z,z)
        prior_var = np.diag(kern.gram(Z_fine, Z_fine))
        results[name] = (H_rec, var_H, prior_var, rmse)
        print(f"    RMSE = {rmse:.6f},  offset = {offset:.4f}")

    # ── Posterior NLL of the true H under each model ─────────────────────
    # NLL = Σ_i [ ½ log(2π σ²_i) + (H_true_i − μ_i)² / (2σ²_i) ]
    # This is a proper scoring rule: penalises both mean error AND
    # miscalibrated variance (too wide or too narrow).
    print("\n  Posterior NLL of true H:")
    for name in ['NKN', 'RBF']:
        H_rec, var_H, _, rmse = results[name][:4]
        sigma2 = np.clip(var_H, 1e-10, None)
        h_true_flat = H_true.ravel()
        h_rec_flat = H_rec.ravel()
        nll_per_point = 0.5 * np.log(2 * np.pi * sigma2) + (h_true_flat - h_rec_flat)**2 / (2 * sigma2)
        total_nll = np.sum(nll_per_point)
        mean_nll = np.mean(nll_per_point)
        print(f"    {name}:  total = {total_nll:.1f},  per-point = {mean_nll:.4f}")
        results[name] = (*results[name], total_nll)  # append NLL

    # ═════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4: Generating figures")
    print("=" * 60)

    # ── PDF evolution ────────────────────────────────────────────────────
    z0_list = [
        (np.array([0.1, 0.2]),  'Near saddle'),
        (np.array([-1.0, 0.0]), 'Left well'),
        (np.array([0.0, 0.9]),  'Outside separatrix'),
    ]
    ensemble_data = []
    for z0, label in z0_list:
        times, ens = evolve_ensemble(z0, PDF_SIGMA, PDF_N_SAMPLES,
                                     PDF_T, PDF_N_SAVE, seed=0)
        ensemble_data.append((times, ens, label))

    n_starts = len(z0_list)
    fig, axes = plt.subplots(n_starts, PDF_N_SAVE, figsize=(4*PDF_N_SAVE, 4*n_starts))
    for row, (times, ens, label) in enumerate(ensemble_data):
        for col in range(PDF_N_SAVE):
            ax = axes[row, col]
            qq = np.linspace(-2.2, 2.2, 200); pp = np.linspace(-1.5, 1.5, 200)
            QQ, PP = np.meshgrid(qq, pp)
            ax.contour(QQ, PP, hamiltonian(QQ, PP), levels=[0],
                       colors='red', linewidths=1, linestyles='--')
            ax.contour(QQ, PP, hamiltonian(QQ, PP),
                       levels=np.linspace(-0.25, 1.0, 12),
                       colors='gray', linewidths=0.3, alpha=0.3)
            ax.scatter(ens[col, :, 0], ens[col, :, 1], s=1, alpha=0.3, c='steelblue')
            ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
            if row == 0: ax.set_title(f't = {times[col]:.2f}', fontsize=11)
            if col == 0: ax.set_ylabel(label, fontsize=11)
            ax.tick_params(labelsize=7)
    fig.suptitle('Phase-Space PDF Evolution', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/pdf_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print("  → pdf_evolution.png")

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON 1+2: Prior and Posterior variance — individual panels + table
    # ══════════════════════════════════════════════════════════════════════
    pv_rbf = results['RBF'][:4][2].reshape(fine_shape)
    pv_nkn = results['NKN'][:4][2].reshape(fine_shape)
    post_rbf = np.clip(results['RBF'][:4][1], 1e-12, None).reshape(fine_shape)
    post_nkn = np.clip(results['NKN'][:4][1], 1e-12, None).reshape(fine_shape)

    variance_panels = [
        ('prior_variance_rbf',  pv_rbf,   'RBF Prior k(z,z)'),
        ('prior_variance_nkn',  pv_nkn,   'NKN Prior k(z,z)'),
        ('posterior_variance_rbf', post_rbf, r'RBF Posterior $\sigma^2$[H]'),
        ('posterior_variance_nkn', post_nkn, r'NKN Posterior $\sigma^2$[H]'),
    ]

    stats = {}
    for fname, data, title in variance_panels:
        fig, ax = plt.subplots(figsize=(7, 5.5))

        # Detect near-constant panels (e.g. RBF prior, which is exactly
        # flat up to floating-point noise) and pad the color range so
        # the true value is centered and visibly labeled, rather than
        # letting LogNorm stretch machine-epsilon noise across the
        # full color scale.
        data_mean = data.mean()
        relative_spread = (data.max() - data.min()) / max(abs(data_mean), 1e-300)

        if relative_spread < 1e-6:
            # Effectively constant: pad ±5% around the true value so
            # it renders as one solid, correctly-labeled color.
            vmin = data_mean * 0.90
            vmax = data_mean * 1.10
            pcm = ax.pcolormesh(FQ, FP, np.full_like(data, data_mean),
                                shading='auto', cmap='inferno',
                                norm=LogNorm(vmin=vmin, vmax=vmax))
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.85, label='Variance')
            # Force a tick exactly at the true constant value
            cbar.set_ticks([data_mean])
            cbar.set_ticklabels([f'{data_mean:.4e}'])
        else:
            vmin = max(data[data > 0].min(), 1e-12)
            vmax = data.max()
            pcm = ax.pcolormesh(FQ, FP, data, shading='auto', cmap='inferno',
                                norm=LogNorm(vmin=vmin, vmax=vmax))
            fig.colorbar(pcm, ax=ax, shrink=0.85, label='Variance')

        ax.contour(FQ, FP, H_true, levels=[0], colors='cyan',
                   linewidths=1.5, linestyles='--')
        ax.scatter(Z_est[:, 0], Z_est[:, 1], s=6, c='white', alpha=0.4, zorder=5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('q'); ax.set_ylabel('p')
        fig.tight_layout()
        fig.savefig(f'{FIGDIR}/{fname}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        stats[fname] = {
            'mean': data.mean(), 'min': data.min(), 'max': data.max(),
            'std': data.std()
        }

    print("  → prior_variance_rbf.png, prior_variance_nkn.png")
    print("  → posterior_variance_rbf.png, posterior_variance_nkn.png")

    # ── LaTeX table ──────────────────────────────────────────────────────
    tex_table = r"""
\begin{table}[h]
\centering
\caption{Variance statistics: RBF vs NKN (prior and posterior)}
\begin{tabular}{l c c c c}
\hline
 & Mean & Min & Max & Std \\
\hline
RBF Prior $k(z,z)$ & %.4e & %.4e & %.4e & %.4e \\
NKN Prior $k(z,z)$ & %.4e & %.4e & %.4e & %.4e \\
\hline
RBF Posterior $\sigma^2[H]$ & %.4e & %.4e & %.4e & %.4e \\
NKN Posterior $\sigma^2[H]$ & %.4e & %.4e & %.4e & %.4e \\
\hline
Ratio (RBF/NKN) Post.\ Mean & \multicolumn{4}{c}{%.1f$\times$} \\
\hline
\end{tabular}
\end{table}
""" % (
        stats['prior_variance_rbf']['mean'], stats['prior_variance_rbf']['min'],
        stats['prior_variance_rbf']['max'], stats['prior_variance_rbf']['std'],
        stats['prior_variance_nkn']['mean'], stats['prior_variance_nkn']['min'],
        stats['prior_variance_nkn']['max'], stats['prior_variance_nkn']['std'],
        stats['posterior_variance_rbf']['mean'], stats['posterior_variance_rbf']['min'],
        stats['posterior_variance_rbf']['max'], stats['posterior_variance_rbf']['std'],
        stats['posterior_variance_nkn']['mean'], stats['posterior_variance_nkn']['min'],
        stats['posterior_variance_nkn']['max'], stats['posterior_variance_nkn']['std'],
        stats['posterior_variance_rbf']['mean'] / max(stats['posterior_variance_nkn']['mean'], 1e-15),
    )
    with open(f'{FIGDIR}/variance_table.tex', 'w') as f:
        f.write(tex_table)
    print("  → variance_table.tex")
    print(tex_table)

    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/variance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → variance_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON 3: Posterior H (top) + Error maps (bottom)
    # ══════════════════════════════════════════════════════════════════════
    levels = np.linspace(-0.3, 1.5, 20)
    H_rbf = results['RBF'][:4][0]; H_nkn = results['NKN'][:4][0]
    rmse_rbf = results['RBF'][:4][3]; rmse_nkn = results['NKN'][:4][3]
    nll_rbf = results['RBF'][4]; nll_nkn = results['NKN'][4]
    err_rbf = H_rbf - H_true; err_nkn = H_nkn - H_true
    elim = max(abs(err_rbf).max(), abs(err_nkn).max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for col, (H_rec, name, rmse, nll) in enumerate([
            (H_rbf, 'RBF', rmse_rbf, nll_rbf),
            (H_nkn, 'NKN', rmse_nkn, nll_nkn),
            (H_true, 'True H', 0, 0)]):
        ax = axes[0, col]
        cs = ax.contourf(FQ, FP, H_rec, levels=levels, cmap='viridis')
        ax.contour(FQ, FP, H_rec, levels=[0], colors='red', linewidths=2)
        fig.colorbar(cs, ax=ax, shrink=0.85)
        if name == 'True H':
            ax.set_title('True H', fontsize=12)
        else:
            ax.set_title(f'{name} (RMSE={rmse:.4f}, NLL={nll:.0f})', fontsize=11)
        ax.set_xlabel('q'); ax.set_ylabel('p')

    el = np.linspace(-elim, elim, 20)
    for col, (err, name) in enumerate([(err_rbf, 'RBF'), (err_nkn, 'NKN')]):
        ax = axes[1, col]
        cs = ax.contourf(FQ, FP, err, levels=el, cmap='RdBu_r')
        fig.colorbar(cs, ax=ax, shrink=0.85, label='error')
        ax.contour(FQ, FP, H_true, levels=[0], colors='black', linewidths=1, linestyles='--')
        ax.set_title(f'{name} Error', fontsize=12)
        ax.set_xlabel('q'); ax.set_ylabel('p')

    ax = axes[1, 2]
    diff = np.abs(err_rbf) - np.abs(err_nkn)
    dl = max(abs(diff).max(), 0.01)
    cs = ax.contourf(FQ, FP, diff, levels=np.linspace(-dl, dl, 20), cmap='RdBu_r')
    fig.colorbar(cs, ax=ax, shrink=0.85, label='|err_RBF|−|err_NKN|')
    ax.contour(FQ, FP, H_true, levels=[0], colors='black', linewidths=1, linestyles='--')
    ax.set_title('|RBF err| − |NKN err| (red=NKN better)', fontsize=10)
    ax.set_xlabel('q'); ax.set_ylabel('p')

    fig.suptitle('Hamiltonian Reconstruction + Error Maps', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/H_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print("  → H_comparison.png")

    # ── NKN structure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(FQ, FP, H_true, levels=[0], colors='red', linewidths=1, linestyles='--')
    ax.contour(FQ, FP, H_true, levels=np.linspace(-0.25, 1.0, 12),
               colors='gray', linewidths=0.3, alpha=0.3)
    amps = np.exp(nkn.log_lam)
    for i in range(nkn.M):
        center, w, h, angle = nkn.get_ellipse_params(i)
        alpha_i = max(amps[i] / max(amps.max(), 1e-12), 0.1)
        e = Ellipse(xy=center, width=w, height=h, angle=angle,
                    fill=False, edgecolor='dodgerblue', linewidth=1.5, alpha=alpha_i)
        ax.add_patch(e)
    sc = ax.scatter(nkn.centers[:, 0], nkn.centers[:, 1],
                    c=amps, s=60, cmap='plasma', edgecolors='k', zorder=5)
    plt.colorbar(sc, ax=ax, label='Amplitude λ_i')
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.5, 1.5)
    ax.set_title('NKN: Anisotropic Inducing Points', fontsize=13)
    ax.set_xlabel('q'); ax.set_ylabel('p')
    fig.tight_layout()
    fig.savefig(f'{FIGDIR}/nkn_structure.png', dpi=150); plt.close(fig)
    print("  → nkn_structure.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name in ['RBF', 'NKN']:
        r = results[name]
        rmse = r[3]
        nll = r[4] if len(r) > 4 else float('nan')
        print(f"  {name}:  RMSE = {rmse:.6f}  NLL(true H) = {nll:.1f}")
    print(f"  RBF params: {rbf}")
    print(f"  NKN params: {nkn}")
    print(f"  Noise: empirical (not fitted)")
    print(f"  Total time: {time.time()-t_start:.0f}s")
    print(f"  Figures in {FIGDIR}")


if __name__ == '__main__':
    print("=" * 60)
    print("  Joint Gradient+Hessian GP — Hamiltonian Reconstruction")
    print("=" * 60 + "\n")
    main()