"""
hessian_gp.py — Structured GP: prior on H(z), observe ∇²H(z) + noise.

Instead of three independent GPs on (H_qq, H_qp, H_pp), we place ONE
GP prior on the scalar Hamiltonian H(z) and observe its Hessian entries.

    H ~ GP(0, k(z,z'))

    Cov[∂²H/∂z_a∂z_b(z),  ∂²H/∂z_c∂z_d(z')]
        = ∂⁴k(z,z') / (∂z_a ∂z_b ∂z'_c ∂z'_d)

    Cov[H(z*),  ∂²H/∂z_c∂z_d(z')]
        = ∂²k(z*,z') / (∂z'_c ∂z'_d)

This automatically enforces:
  - Hessian symmetry (H_qp = H_pq by construction)
  - Integrability (all observations are derivatives of one function)
  - H is directly recoverable from the posterior, no integration needed

Hessian indices: we use the 3 unique entries (qq, qp, pp) → indices 0,1,2.
The index pairs are: 0→(0,0), 1→(0,1), 2→(1,1).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# Map from flat index (0,1,2) to (a,b) pairs
IDX_PAIRS = [(0, 0), (0, 1), (1, 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel derivative computations
# ═══════════════════════════════════════════════════════════════════════════════

def nkn_d4(kernel, z, zp, a, b, c, d):
    """
    ∂⁴k_NKN / (∂z_a ∂z_b ∂z'_c ∂z'_d)

    For separable NKN: k_i(z,z') = λ_i φ(z−p_i) φ(z'−p_i), so
      ∂⁴k_i/(∂z_a∂z_b∂z'_c∂z'_d) = λ_i · D²φ_ab(z−p) · D²φ_cd(z'−p)

    where D²φ_ab(u) = (u_a u_b/(ℓ_a² ℓ_b²) − δ_{ab}/ℓ_a²) · φ(u).
    """
    total = 0.0
    for i in range(kernel.M):
        lam = np.exp(kernel.log_lam[i])
        ell = np.exp(kernel.log_ell[i])       # (2,)
        p = kernel.centers[i]

        u = z - p
        v = zp - p

        phi_u = np.exp(-0.5 * np.sum((u / ell)**2))
        phi_v = np.exp(-0.5 * np.sum((v / ell)**2))

        # D²φ_ab(u) = (u_a u_b / (ℓ_a² ℓ_b²) − δ_{ab}/ℓ_a²) φ(u)
        D2_ab = (u[a] * u[b] / (ell[a]**2 * ell[b]**2)
                 - (1.0 if a == b else 0.0) / ell[a]**2) * phi_u

        D2_cd = (v[c] * v[d] / (ell[c]**2 * ell[d]**2)
                 - (1.0 if c == d else 0.0) / ell[c]**2) * phi_v

        total += lam * D2_ab * D2_cd
    return total


def nkn_d2_right(kernel, z, zp, c, d):
    """
    ∂²k_NKN / (∂z'_c ∂z'_d)   [derivatives only on the second argument]

    = Σ_i λ_i φ(z−p_i) · D²φ_cd(z'−p_i)

    Used for cross-covariance between H(z*) and Hessian observations.
    """
    total = 0.0
    for i in range(kernel.M):
        lam = np.exp(kernel.log_lam[i])
        ell = np.exp(kernel.log_ell[i])
        p = kernel.centers[i]

        u = z - p
        v = zp - p

        phi_u = np.exp(-0.5 * np.sum((u / ell)**2))
        phi_v = np.exp(-0.5 * np.sum((v / ell)**2))

        D2_cd = (v[c] * v[d] / (ell[c]**2 * ell[d]**2)
                 - (1.0 if c == d else 0.0) / ell[c]**2) * phi_v

        total += lam * phi_u * D2_cd
    return total


def rbf_d4(kernel, z, zp, a, b, c, d):
    """
    ∂⁴k_RBF / (∂z_a ∂z_b ∂z'_c ∂z'_d)

    For k = σ² exp(−‖d‖²/(2ℓ²)), d = z−z':

    = (k/ℓ⁴) [ d_a d_b d_c d_d/ℓ⁴
               − (δ_{ab}d_c d_d + δ_{cd}d_a d_b + δ_{ac}d_b d_d
                  + δ_{ad}d_b d_c + δ_{bc}d_a d_d + δ_{bd}d_a d_c)/ℓ²
               + (δ_{ab}δ_{cd} + δ_{ac}δ_{bd} + δ_{ad}δ_{bc}) ]
    """
    dd = z - zp
    r2 = np.sum(dd**2)
    ell = kernel.ell
    k_val = kernel.sigma_f**2 * np.exp(-r2 / (2 * ell**2))
    ell2 = ell**2

    def delta(i, j):
        return 1.0 if i == j else 0.0

    T = (dd[a]*dd[b]*dd[c]*dd[d] / ell2**2
         - (delta(a,b)*dd[c]*dd[d] + delta(c,d)*dd[a]*dd[b]
            + delta(a,c)*dd[b]*dd[d] + delta(a,d)*dd[b]*dd[c]
            + delta(b,c)*dd[a]*dd[d] + delta(b,d)*dd[a]*dd[c]) / ell2
         + (delta(a,b)*delta(c,d) + delta(a,c)*delta(b,d)
            + delta(a,d)*delta(b,c)))

    return k_val / ell2**2 * T


def rbf_d2_right(kernel, z, zp, c, d):
    """
    ∂²k_RBF / (∂z'_c ∂z'_d) = (k/ℓ²)(d_c d_d/ℓ² − δ_{cd})
    where d = z − z'.
    """
    dd = z - zp
    r2 = np.sum(dd**2)
    ell = kernel.ell
    k_val = kernel.sigma_f**2 * np.exp(-r2 / (2 * ell**2))

    return (k_val / ell**2) * (dd[c]*dd[d] / ell**2
                                - (1.0 if c == d else 0.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Gram matrix construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_hessian_gram(kernel, Z, d4_fn):
    """
    Build the 3N × 3N Gram matrix for Hessian observations.
    Vectorised: computes all (a,b,c,d) blocks in one pass per inducing point.

    D²φ_s(u) = ([Σ^{-1}u]_a [Σ^{-1}u]_b − [Σ^{-1}]_{ab}) φ(u)
    where (a,b) = IDX_PAIRS[s].  Works for full (non-diagonal) Σ^{-1}.
    """
    N = len(Z)
    K = np.zeros((3*N, 3*N))

    if hasattr(kernel, 'log_lam'):
        for ii in range(kernel.M):
            lam = np.exp(kernel.log_lam[ii])
            Sinv = kernel._get_Sinv(ii)
            p = kernel.centers[ii]

            U = Z - p                                     # (N, 2)
            SU = U @ Sinv                                  # (N, 2) = Σ^{-1}u per row
            phi = np.exp(-0.5 * np.sum(U * SU, axis=1))   # (N,)

            # D2[n, s] = ([Sinv u]_a [Sinv u]_b − Sinv_{ab}) φ(u_n)
            D2 = np.zeros((N, 3))
            for s, (a, b) in enumerate(IDX_PAIRS):
                D2[:, s] = (SU[:, a] * SU[:, b] - Sinv[a, b]) * phi

            for s in range(3):
                for t in range(3):
                    K[s::3, t::3] += lam * np.outer(D2[:, s], D2[:, t])
    else:
        for i in range(N):
            for j in range(N):
                for s in range(3):
                    a, b = IDX_PAIRS[s]
                    for t in range(3):
                        c, d = IDX_PAIRS[t]
                        K[3*i+s, 3*j+t] = d4_fn(kernel, Z[i], Z[j], a,b,c,d)
    return K


def build_H_cross(kernel, Z_test, Z_train, d2_fn):
    """Cross-covariance between H(z*) and Hessian obs. Vectorised for NKN."""
    M_test = len(Z_test)
    N = len(Z_train)
    K_cross = np.zeros((M_test, 3*N))

    if hasattr(kernel, 'log_lam'):
        for ii in range(kernel.M):
            lam = np.exp(kernel.log_lam[ii])
            Sinv = kernel._get_Sinv(ii)
            p = kernel.centers[ii]

            U_test = Z_test - p
            U_train = Z_train - p

            SU_test = U_test @ Sinv
            SU_train = U_train @ Sinv

            phi_test = np.exp(-0.5 * np.sum(U_test * SU_test, axis=1))
            phi_train = np.exp(-0.5 * np.sum(U_train * SU_train, axis=1))

            D2_train = np.zeros((N, 3))
            for s, (c, d) in enumerate(IDX_PAIRS):
                D2_train[:, s] = (SU_train[:, c] * SU_train[:, d]
                                  - Sinv[c, d]) * phi_train

            for t in range(3):
                K_cross[:, t::3] += lam * np.outer(phi_test, D2_train[:, t])
    else:
        for m in range(M_test):
            for j in range(N):
                for t in range(3):
                    c, d = IDX_PAIRS[t]
                    K_cross[m, 3*j+t] = d2_fn(kernel, Z_test[m], Z_train[j], c, d)
    return K_cross


# ═══════════════════════════════════════════════════════════════════════════════
# Fitting and prediction
# ═══════════════════════════════════════════════════════════════════════════════

def stack_hessian_obs(Hqq, Hqp, Hpp):
    """Stack N Hessian observations into a 3N vector."""
    N = len(Hqq)
    y = np.zeros(3*N)
    for i in range(N):
        y[3*i + 0] = Hqq[i]
        y[3*i + 1] = Hqp[i]
        y[3*i + 2] = Hpp[i]
    return y


def hessian_nll(kernel, Z, y_hess, d4_fn, jitter=1e-6):
    """Negative log marginal likelihood for Hessian observations."""
    N3 = len(y_hess)
    K = build_hessian_gram(kernel, Z, d4_fn)
    K += (kernel.sigma_n**2 + jitter) * np.eye(N3)

    try:
        L, lower = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = cho_solve((L, lower), y_hess)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * y_hess @ alpha + 0.5 * log_det + 0.5 * N3 * np.log(2*np.pi)


def hessian_fit(kernel, Z, y_hess, d4_fn, max_iter=200, verbose=True):
    """Optimise kernel hyperparameters for Hessian observations."""
    p0 = kernel.get_params()
    calls = [0]

    def obj(params):
        kernel.set_params(params)
        v = hessian_nll(kernel, Z, y_hess, d4_fn)
        calls[0] += 1
        if verbose and calls[0] % 50 == 0:
            print(f"      iter {calls[0]:5d}  NLL={v:.2f}")
        return v

    res = minimize(obj, p0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-9})
    kernel.set_params(res.x)
    if verbose:
        print(f"      Done: NLL={res.fun:.2f} ({res.success})")
    return res


def predict_H(kernel, Z_train, y_hess, Z_test, d4_fn, d2_fn, jitter=1e-6):
    """
    Posterior mean and variance of H(z*) given Hessian observations.

    Returns
    -------
    mu_H  : (M,) — posterior mean of H at test points
    var_H : (M,) — posterior marginal variance
    """
    N = len(Z_train)
    N3 = 3 * N
    K = build_hessian_gram(kernel, Z_train, d4_fn)
    K += (kernel.sigma_n**2 + jitter) * np.eye(N3)
    L, lower = cho_factor(K, lower=True)
    alpha = cho_solve((L, lower), y_hess)

    # Cross-covariance: Cov[H(z*), Hessian obs]
    K_cross = build_H_cross(kernel, Z_test, Z_train, d2_fn)  # (M, 3N)
    mu_H = K_cross @ alpha

    # Prior variance of H: k(z*, z*)
    M = len(Z_test)
    k_diag = np.array([kernel.gram(Z_test[m:m+1], Z_test[m:m+1])[0, 0]
                        for m in range(M)])

    V = cho_solve((L, lower), K_cross.T)  # (3N, M)
    var_H = k_diag - np.sum(K_cross.T * V, axis=0)

    return mu_H, np.clip(var_H, 0, None)