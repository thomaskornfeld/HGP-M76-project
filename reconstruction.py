"""
reconstruction.py — Reconstruct H(q,p) from its Hessian field ∇²H.

Given H_qq(q,p), H_qp(q,p), H_pp(q,p) on a regular grid, integrate
twice to recover H up to an affine ambiguity, anchored at the saddle
point (q_ref, p_ref) with H = 0 and ∇H = 0.

Path:  integrate along q first (at p = p_ref), then along p.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def reconstruct(q_grid, p_grid, Hqq, Hqp, Hpp, q_ref=0.0, p_ref=0.0):
    """
    Parameters
    ----------
    q_grid, p_grid : 1-D arrays of grid coordinates
    Hqq, Hqp, Hpp  : 2-D arrays, shape (len(p_grid), len(q_grid))
        Hessian components on the meshgrid.
    q_ref, p_ref    : float, anchor point (∇H = 0, H = 0 here)

    Returns
    -------
    H_recon : 2-D array, shape (len(p_grid), len(q_grid))
    """
    nq = len(q_grid)
    np_ = len(p_grid)

    # Find indices closest to the reference point
    iq0 = np.argmin(np.abs(q_grid - q_ref))
    ip0 = np.argmin(np.abs(p_grid - p_ref))

    # ─── Step 1: ∂H/∂q along the line p = p_ref ────────────────────────
    # Hq(q, p_ref) = 0 + ∫_{q_ref}^{q} Hqq(q', p_ref) dq'
    Hq = np.zeros((np_, nq))
    row_qq = Hqq[ip0, :]
    # Integrate rightward from iq0
    if iq0 < nq - 1:
        Hq[ip0, iq0+1:] = cumulative_trapezoid(row_qq[iq0:], q_grid[iq0:])
    # Integrate leftward from iq0
    if iq0 > 0:
        Hq[ip0, :iq0] = -cumulative_trapezoid(row_qq[iq0::-1],
                                                q_grid[iq0::-1])[::-1]

    # ─── Step 2: extend Hq to all p via ∂Hq/∂p = Hqp ───────────────────
    for iq in range(nq):
        col_qp = Hqp[:, iq]
        if ip0 < np_ - 1:
            Hq[ip0+1:, iq] = Hq[ip0, iq] + cumulative_trapezoid(
                col_qp[ip0:], p_grid[ip0:])
        if ip0 > 0:
            Hq[:ip0, iq] = Hq[ip0, iq] - cumulative_trapezoid(
                col_qp[ip0::-1], p_grid[ip0::-1])[::-1]

    # ─── Step 3: ∂H/∂p along the line p = p_ref ────────────────────────
    # Hp(q, p_ref) = 0 + ∫_{q_ref}^{q} Hqp(q', p_ref) dq'
    Hp = np.zeros((np_, nq))
    row_qp = Hqp[ip0, :]
    if iq0 < nq - 1:
        Hp[ip0, iq0+1:] = cumulative_trapezoid(row_qp[iq0:], q_grid[iq0:])
    if iq0 > 0:
        Hp[ip0, :iq0] = -cumulative_trapezoid(row_qp[iq0::-1],
                                                q_grid[iq0::-1])[::-1]

    # ─── Step 4: extend Hp to all p via ∂Hp/∂p = Hpp ────────────────────
    for iq in range(nq):
        col_pp = Hpp[:, iq]
        if ip0 < np_ - 1:
            Hp[ip0+1:, iq] = Hp[ip0, iq] + cumulative_trapezoid(
                col_pp[ip0:], p_grid[ip0:])
        if ip0 > 0:
            Hp[:ip0, iq] = Hp[ip0, iq] - cumulative_trapezoid(
                col_pp[ip0::-1], p_grid[ip0::-1])[::-1]

    # ─── Step 5: H along p = p_ref ──────────────────────────────────────
    H = np.zeros((np_, nq))
    row_Hq = Hq[ip0, :]
    if iq0 < nq - 1:
        H[ip0, iq0+1:] = cumulative_trapezoid(row_Hq[iq0:], q_grid[iq0:])
    if iq0 > 0:
        H[ip0, :iq0] = -cumulative_trapezoid(row_Hq[iq0::-1],
                                               q_grid[iq0::-1])[::-1]

    # ─── Step 6: extend H to all p via ∂H/∂p = Hp ───────────────────────
    for iq in range(nq):
        col_Hp = Hp[:, iq]
        if ip0 < np_ - 1:
            H[ip0+1:, iq] = H[ip0, iq] + cumulative_trapezoid(
                col_Hp[ip0:], p_grid[ip0:])
        if ip0 > 0:
            H[:ip0, iq] = H[ip0, iq] - cumulative_trapezoid(
                col_Hp[ip0::-1], p_grid[ip0::-1])[::-1]

    return H
