"""
hamiltonian_gp.py — Joint GP on H, conditioning on gradient AND Hessian.

Observation vector at each point z_i (5 components):
    h_i = (∂qH, ∂pH, ∂q²H, ∂q∂pH, ∂p²H)

These are rows 1–5 of the 6-vector F(z) = (H, ∂qH, ∂pH, ∂q²H, ∂q∂pH, ∂p²H),
all derivatives of one scalar GP H ~ GP(0, k).

The 5N × 5N Gram matrix uses derivatives of k up to 4th order.
For the NKN's separable structure, every derivative factors:
    ∂^α_z ∂^β_{z'} k_r(z,z') = λ_r · D^α φ_r(z−p_r) · D^β φ_r(z'−p_r)

The 5 derivative operators D^s on φ(u) with w := Σ^{-1}u are:
    s=0  ∂q    :  −w₀ φ
    s=1  ∂p    :  −w₁ φ
    s=2  ∂q²   :  (w₀² − Σ⁻¹₀₀) φ
    s=3  ∂q∂p  :  (w₀w₁ − Σ⁻¹₀₁) φ
    s=4  ∂p²   :  (w₁² − Σ⁻¹₁₁) φ

Two noise levels: σ_grad for gradient obs, σ_hess for Hessian obs.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

N_OBS = 5  # observations per point


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised Gram and cross-covariance builders (NKN-specific fast path)
# ═══════════════════════════════════════════════════════════════════════════════

def _nkn_deriv_matrix(kernel, Z):
    """
    Precompute the N × 5 × M derivative array for all points and all
    inducing bumps.  D[n, s, r] = D^s φ_r(z_n − p_r).

    Returns D (N, 5, M) and phi (N, M).
    """
    N = len(Z)
    M = kernel.M
    Ds = np.zeros((N, N_OBS, M))
    phi = np.zeros((N, M))

    for r in range(M):
        Sinv = kernel._get_Sinv(r)
        p = kernel.centers[r]
        U = Z - p                                    # (N, 2)
        W = U @ Sinv                                  # (N, 2)  w = Σ⁻¹u
        ph = np.exp(-0.5 * np.sum(U * W, axis=1))    # (N,)
        phi[:, r] = ph

        # s=0: ∂q  →  −w₀ φ
        Ds[:, 0, r] = -W[:, 0] * ph
        # s=1: ∂p  →  −w₁ φ
        Ds[:, 1, r] = -W[:, 1] * ph
        # s=2: ∂q² →  (w₀² − Σ⁻¹₀₀) φ
        Ds[:, 2, r] = (W[:, 0]**2 - Sinv[0, 0]) * ph
        # s=3: ∂q∂p → (w₀w₁ − Σ⁻¹₀₁) φ
        Ds[:, 3, r] = (W[:, 0] * W[:, 1] - Sinv[0, 1]) * ph
        # s=4: ∂p² →  (w₁² − Σ⁻¹₁₁) φ
        Ds[:, 4, r] = (W[:, 1]**2 - Sinv[1, 1]) * ph

    return Ds, phi


def _hermite(n, x):
    """Probabilist's Hermite polynomial He_n(x)."""
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    if n == 2: return x**2 - 1
    if n == 3: return x**3 - 3*x
    if n == 4: return x**4 - 6*x**2 + 3
    raise ValueError(f"Hermite order {n} not implemented")


# Multi-indices for the 5 observation types: (m_q, m_p)
_MULTI = [(1,0), (0,1), (2,0), (1,1), (0,2)]
_ORDER = [1, 1, 2, 2, 2]  # |α_s|


def _rbf_gram_block(kernel, Z1, Z2):
    """
    Analytic 5×5 derivative block for the RBF kernel, vectorised over
    all (i,j) pairs.  Returns shape (5*N1, 5*N2).

    Uses: D^α_z D'^β_{z'} k = (-1)^|β| D^{α+β} k
    and:  D_q^m D_p^n k = (-1)^{m+n}/ℓ^{m+n} He_m(u) He_n(v) k
    """
    sf2 = kernel.sigma_f**2
    ell = kernel.ell
    N1, N2 = len(Z1), len(Z2)

    # Pairwise displacements
    Dq = Z1[:, 0:1] - Z2[:, 0:1].T   # (N1, N2)
    Dp = Z1[:, 1:2] - Z2[:, 1:2].T   # (N1, N2)
    U = Dq / ell
    V = Dp / ell
    K_base = sf2 * np.exp(-0.5 * (U**2 + V**2))   # (N1, N2)

    K = np.zeros((5*N1, 5*N2))

    for s in range(5):
        ms, ns = _MULTI[s]
        for t in range(5):
            mt, nt = _MULTI[t]

            # Combined multi-index for D^{α_s + α_t}
            m_tot = ms + mt
            n_tot = ns + nt
            order_tot = m_tot + n_tot

            # D^{α_s + α_t} k = (-1)^{order_tot} / ℓ^{order_tot} He_m(u) He_n(v) k
            # Multiply by (-1)^{|α_t|} for the z' derivatives
            sign = (-1)**_ORDER[t] * (-1)**order_tot
            scale = 1.0 / ell**order_tot

            block = sign * scale * _hermite(m_tot, U) * _hermite(n_tot, V) * K_base
            K[s::5, t::5] = block

    return K


def _rbf_cross_block(kernel, Z_test, Z_train):
    """
    Analytic cross-covariance Cov[H(z*), D'^α_t H(z_j)] for the RBF.
    = D'^α_t k(z*, z_j) = (-1)^{|α_t|} D^{α_t} k = He_m(u) He_n(v) k / ℓ^|α_t|
    (the double sign cancels).  Returns (M_test, 5*N_train).
    """
    sf2 = kernel.sigma_f**2
    ell = kernel.ell
    Mt, N = len(Z_test), len(Z_train)

    Dq = Z_test[:, 0:1] - Z_train[:, 0:1].T
    Dp = Z_test[:, 1:2] - Z_train[:, 1:2].T
    U = Dq / ell
    V = Dp / ell
    K_base = sf2 * np.exp(-0.5 * (U**2 + V**2))

    K_cross = np.zeros((Mt, 5*N))
    for t in range(5):
        mt, nt = _MULTI[t]
        order = _ORDER[t]
        # (-1)^|α_t| · (-1)^|α_t| = 1, so no sign
        scale = 1.0 / ell**order
        K_cross[:, t::5] = scale * _hermite(mt, U) * _hermite(nt, V) * K_base

    return K_cross


def build_gram(kernel, Z):
    """
    Build the 5N × 5N Gram matrix for joint gradient+Hessian observations.
    NKN: fast vectorised path.  RBF/other: numerical derivatives.
    """
    N = len(Z)
    K = np.zeros((N_OBS * N, N_OBS * N))

    if hasattr(kernel, 'log_lam'):
        # NKN fast path
        Ds, _ = _nkn_deriv_matrix(kernel, Z)
        lam = np.exp(kernel.log_lam)
        for s in range(N_OBS):
            for t in range(N_OBS):
                A = Ds[:, s, :] * lam[np.newaxis, :]
                B = Ds[:, t, :]
                K[s::N_OBS, t::N_OBS] = A @ B.T
    else:
        # RBF: analytic Hermite-polynomial derivatives (exact, PSD by construction)
        K = _rbf_gram_block(kernel, Z, Z)
    K = 0.5 * (K + K.T)
    return K


def build_H_cross(kernel, Z_test, Z_train):
    """
    Cross-covariance between H(z*) and the 5N observation vector.
    NKN: fast path.  RBF: analytic Hermite path.
    Returns (M_test, 5*N_train).
    """
    M_test = len(Z_test)
    N_train = len(Z_train)
    K_cross = np.zeros((M_test, N_OBS * N_train))

    if hasattr(kernel, 'log_lam'):
        Ds_train, _ = _nkn_deriv_matrix(kernel, Z_train)
        _, phi_test = _nkn_deriv_matrix(kernel, Z_test)
        lam = np.exp(kernel.log_lam)
        for t in range(N_OBS):
            A = phi_test * lam[np.newaxis, :]
            B = Ds_train[:, t, :]
            K_cross[:, t::N_OBS] = A @ B.T
    else:
        # RBF: analytic cross-covariance
        K_cross = _rbf_cross_block(kernel, Z_test, Z_train)

    return K_cross


# ═══════════════════════════════════════════════════════════════════════════════
# Noise diagonal (block-structured: σ_grad for rows 0-1, σ_hess for rows 2-4)
# ═══════════════════════════════════════════════════════════════════════════════

def build_noise_matrix(noise_grad, noise_hess):
    """
    Build the 5N × 5N block-diagonal noise matrix from per-point
    empirical covariances.

    noise_grad : (N, 2, 2) — gradient noise covariance at each point
    noise_hess : (N, 3, 3) — Hessian noise covariance at each point

    Returns a 5N × 5N matrix, block-diagonal with 5×5 blocks:
        [[grad_2x2,  0      ],
         [0,         hess_3x3]]
    """
    N = len(noise_grad)
    D = np.zeros((N_OBS * N, N_OBS * N))
    for i in range(N):
        idx = N_OBS * i
        D[idx:idx+2, idx:idx+2] = noise_grad[i]
        D[idx+2:idx+5, idx+2:idx+5] = noise_hess[i]
    return D


# Keep the old scalar version as a fallback
def noise_diag(N, sigma_grad, sigma_hess):
    """Build the 5N noise diagonal with two separate noise levels."""
    d = np.zeros(N_OBS * N)
    for i in range(N):
        d[N_OBS*i + 0] = sigma_grad**2
        d[N_OBS*i + 1] = sigma_grad**2
        d[N_OBS*i + 2] = sigma_hess**2
        d[N_OBS*i + 3] = sigma_hess**2
        d[N_OBS*i + 4] = sigma_hess**2
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Stack observations
# ═══════════════════════════════════════════════════════════════════════════════

def stack_observations(grad_H, Hqq, Hqp, Hpp):
    """
    Stack gradient + Hessian observations into the 5N vector y.

    grad_H : (N, 2)  — estimated (∂qH, ∂pH) at each point
    Hqq, Hqp, Hpp : (N,) — estimated Hessian components
    """
    N = len(Hqq)
    y = np.zeros(N_OBS * N)
    for i in range(N):
        y[N_OBS*i + 0] = grad_H[i, 0]
        y[N_OBS*i + 1] = grad_H[i, 1]
        y[N_OBS*i + 2] = Hqq[i]
        y[N_OBS*i + 3] = Hqp[i]
        y[N_OBS*i + 4] = Hpp[i]
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# Fitting
# ═══════════════════════════════════════════════════════════════════════════════

def joint_nll_empirical(kernel, Z, y, noise_matrix, jitter=1e-6):
    """NLL with precomputed per-point noise matrix (no noise params to fit)."""
    K = build_gram(kernel, Z) + noise_matrix + jitter * np.eye(len(y))
    try:
        L, lower = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = cho_solve((L, lower), y)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(y) * np.log(2*np.pi)


def joint_fit_empirical(kernel, Z, y, noise_matrix, max_iter=300,
                        bounds=None, verbose=True):
    """Fit kernel params only — noise is fixed from empirical estimates."""
    p0 = kernel.get_params()
    calls = [0]
    def obj(params):
        kernel.set_params(params)
        v = joint_nll_empirical(kernel, Z, y, noise_matrix)
        calls[0] += 1
        if verbose and calls[0] % 50 == 0:
            print(f"      iter {calls[0]:5d}  NLL={v:.2f}")
        return v
    res = minimize(obj, p0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'maxfun': max(15000, max_iter*50), 'ftol': 1e-9})
    kernel.set_params(res.x)
    if verbose:
        print(f"      Done: NLL={res.fun:.2f} ({res.success})")
    return res


def predict_grad_empirical(kernel, Z_train, y, Z_test, noise_matrix,
                           component, jitter=1e-4):
    """
    Posterior mean and variance of ∂H/∂z_component at test points.

    component=0 → ∂qH (= -ṗ),  component=1 → ∂pH (= q̇).

    Cross-cov: Cov[D_s H(z*), D_t H(z_j)] = Σ_r λ_r D_s φ_r(z*-p_r) D_t φ_r(z_j-p_r)
    where s indexes the gradient component (0 or 1) at the test point.
    """
    N = len(Z_train)
    K = build_gram(kernel, Z_train) + noise_matrix + jitter * np.eye(len(y))

    try:
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y)
        solve_fn = lambda b: cho_solve((L, lower), b)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-6)
        alpha = eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ y))
        solve_fn = lambda b: eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ b))

    M_test = len(Z_test)

    if hasattr(kernel, 'log_lam'):
        # NKN fast path
        Ds_train, _ = _nkn_deriv_matrix(kernel, Z_train)
        Ds_test, _ = _nkn_deriv_matrix(kernel, Z_test)
        lam = np.exp(kernel.log_lam)

        # Cross-cov: Cov[D_component H(z*), D_t H(z_j)]
        # = Σ_r λ_r D_component φ_r(z*-p_r) D_t φ_r(z_j-p_r)
        K_cross = np.zeros((M_test, N_OBS * N))
        for t in range(N_OBS):
            A = Ds_test[:, component, :] * lam[np.newaxis, :]
            B = Ds_train[:, t, :]
            K_cross[:, t::N_OBS] = A @ B.T

        # Prior variance of ∂_component H: k_{ss}(z*,z*)
        # = Σ_r λ_r [D_component φ_r(z*-p_r)]²
        k_diag = np.sum(Ds_test[:, component, :]**2 * lam[np.newaxis, :], axis=1)
    else:
        # RBF: use Hermite derivatives
        # Prior var of ∂_a H = ∂²k/∂z_a∂z'_a|_{z=z'} = σ_f²/ℓ² (always)
        k_diag = np.full(M_test, kernel.sigma_f**2 / kernel.ell**2)

        # Cross-cov via Hermite: D_component^L D_t^R k(z*, z_j)
        # The test-point derivative is first-order (component 0 or 1)
        # Combined with observation type t on the training side
        K_cross = np.zeros((M_test, N_OBS * N))
        for t in range(N_OBS):
            mt_left, nt_left = [(1,0),(0,1)][component]  # test derivative
            mt_right, nt_right = _MULTI[t]                 # train derivative
            m_tot = mt_left + mt_right
            n_tot = nt_left + nt_right
            order_tot = m_tot + n_tot
            sign = (-1)**_ORDER[t] * (-1)**order_tot
            Dq = Z_test[:, 0:1] - Z_train[:, 0:1].T
            Dp = Z_test[:, 1:2] - Z_train[:, 1:2].T
            U = Dq / kernel.ell; V = Dp / kernel.ell
            K_base = kernel.sigma_f**2 * np.exp(-0.5*(U**2+V**2))
            scale = 1.0 / kernel.ell**order_tot
            block = sign * scale * _hermite(m_tot, U) * _hermite(n_tot, V) * K_base
            K_cross[:, t::N_OBS] = block

    mu = K_cross @ alpha
    V = solve_fn(K_cross.T)
    var = k_diag - np.sum(K_cross.T * V, axis=0)
    return mu, np.clip(var, 0, None)
    """Predict H with empirical per-point noise (no fitted noise params)."""
    K = build_gram(kernel, Z_train) + noise_matrix + jitter * np.eye(len(y))
    try:
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y)
        solve_fn = lambda b: cho_solve((L, lower), b)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-6)
        alpha = eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ y))
        solve_fn = lambda b: eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ b))

    K_cross = build_H_cross(kernel, Z_test, Z_train)
    mu_H = K_cross @ alpha

    M_test = len(Z_test)
    k_diag = np.array([kernel.gram(Z_test[m:m+1], Z_test[m:m+1])[0, 0]
                        for m in range(M_test)])
    V = solve_fn(K_cross.T)
    var_H = k_diag - np.sum(K_cross.T * V, axis=0)
    return mu_H, np.clip(var_H, 0, None)
    """NLL for the joint gradient+Hessian observations."""
    N = len(Z)
    K = build_gram(kernel, Z)
    nd = noise_diag(N, sigma_grad, sigma_hess) + jitter
    K += np.diag(nd)

    try:
        L, lower = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = cho_solve((L, lower), y)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    n5 = N_OBS * N
    return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n5 * np.log(2*np.pi)


def joint_fit(kernel, Z, y, max_iter=300, bounds=None, verbose=True):
    """
    Optimise kernel params + two noise levels jointly.

    Appends [log σ_grad, log σ_hess] to the kernel parameter vector.
    """
    p0_kernel = kernel.get_params()
    p0 = np.concatenate([p0_kernel, [np.log(0.1), np.log(0.3)]])
    calls = [0]

    def obj(params):
        kernel.set_params(params[:-2])
        sg = np.exp(params[-2])
        sh = np.exp(params[-1])
        v = joint_nll(kernel, Z, y, sg, sh)
        calls[0] += 1
        if verbose and calls[0] % 50 == 0:
            print(f"      iter {calls[0]:5d}  NLL={v:.2f}  "
                  f"σ_grad={sg:.4f}  σ_hess={sh:.4f}")
        return v

    res = minimize(obj, p0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'maxfun': max(15000, max_iter*50), 'ftol': 1e-9})
    kernel.set_params(res.x[:-2])
    sg_final = np.exp(res.x[-2])
    sh_final = np.exp(res.x[-1])
    if verbose:
        print(f"      Done: NLL={res.fun:.2f}  σ_grad={sg_final:.4f}  "
              f"σ_hess={sh_final:.4f}")
    return res, sg_final, sh_final


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def predict_H(kernel, Z_train, y, Z_test, sigma_grad, sigma_hess, jitter=1e-4):
    """
    Posterior mean and variance of H(z*) given joint gradient+Hessian obs.
    """
    N = len(Z_train)
    K = build_gram(kernel, Z_train)
    nd = noise_diag(N, sigma_grad, sigma_hess) + jitter
    K += np.diag(nd)

    try:
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y)
        solve_fn = lambda b: cho_solve((L, lower), b)
    except np.linalg.LinAlgError:
        # Fallback: eigenvalue clipping for numerical stability
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-6)
        K_inv_y = eigvecs @ (np.diag(1.0 / eigvals) @ (eigvecs.T @ y))
        alpha = K_inv_y
        solve_fn = lambda b: eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ b))

    K_cross = build_H_cross(kernel, Z_test, Z_train)
    mu_H = K_cross @ alpha

    M_test = len(Z_test)
    k_diag = np.array([kernel.gram(Z_test[m:m+1], Z_test[m:m+1])[0, 0]
                        for m in range(M_test)])

    V = solve_fn(K_cross.T)
    var_H = k_diag - np.sum(K_cross.T * V, axis=0)

    return mu_H, np.clip(var_H, 0, None)


def predict_H_empirical(kernel, Z_train, y, Z_test, noise_matrix, jitter=1e-4):
    """Predict H with empirical per-point noise (no fitted noise params)."""
    K = build_gram(kernel, Z_train) + noise_matrix + jitter * np.eye(len(y))
    try:
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y)
        solve_fn = lambda b: cho_solve((L, lower), b)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-6)
        alpha = eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ y))
        solve_fn = lambda b: eigvecs @ (np.diag(1.0/eigvals) @ (eigvecs.T @ b))

    K_cross = build_H_cross(kernel, Z_test, Z_train)
    mu_H = K_cross @ alpha

    M_test = len(Z_test)
    k_diag = np.array([kernel.gram(Z_test[m:m+1], Z_test[m:m+1])[0, 0]
                        for m in range(M_test)])
    V = solve_fn(K_cross.T)
    var_H = k_diag - np.sum(K_cross.T * V, axis=0)
    return mu_H, np.clip(var_H, 0, None)