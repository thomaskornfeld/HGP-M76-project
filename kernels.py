"""
kernels.py — RBF and NKN kernels for GP smoothing of the Hessian field.
"""

import numpy as np


class RBFKernel:
    """k(z,z') = σ_f² exp(−‖z−z'‖²/(2ℓ²)).  Params in log-space."""

    def __init__(self, log_sf=0.0, log_ell=0.0, log_sn=-2.0):
        self.params = np.array([log_sf, log_ell, log_sn])

    @property
    def sigma_f(self):  return np.exp(self.params[0])
    @property
    def ell(self):      return np.exp(self.params[1])
    @property
    def sigma_n(self):  return np.exp(self.params[2])
    @property
    def n_params(self): return 3
    def get_params(self):    return self.params.copy()
    def set_params(self, p): self.params = np.asarray(p, dtype=float)

    def gram(self, Z1, Z2):
        sq = _sq_dist(Z1, Z2).clip(min=0)
        return self.sigma_f**2 * np.exp(-sq / (2 * self.ell**2))

    def __repr__(self):
        return f"RBF(σf={self.sigma_f:.3f}, ℓ={self.ell:.3f}, σn={self.sigma_n:.4f})"


class NKNKernel:
    """
    k(z,z') = Σ_i λ_i φ_i(z−p_i) φ_i(z'−p_i)

    with φ_i(u) = exp(−½ u^T Σ_i^{-1} u),  Σ_i = L_i L_i^T  (full SPD).

    Cholesky parameterisation (guarantees Σ_i is SPD):
        L_i = [[exp(a_i),  0      ],
               [b_i,       exp(c_i)]]

    Per inducing point: log λ_i, a_i, b_i, c_i, p_iq, p_ip  (6 params).
    Plus one global log σ_n.  Total: 6M + 1.
    """

    def __init__(self, centers, log_lam=None, chol_params=None, log_sn=-2.0):
        """
        Parameters
        ----------
        centers     : (M, 2) initial inducing-point positions
        log_lam     : (M,)   initial log-amplitudes
        chol_params : (M, 3) initial Cholesky params [a, b, c] per point
                      If None, defaults to axis-aligned with ℓ=0.5
        """
        self.M = len(centers)
        self.centers = np.array(centers, dtype=float)
        self.log_lam = np.zeros(self.M) if log_lam is None else np.asarray(log_lam, dtype=float)

        if chol_params is None:
            # Default: axis-aligned, ℓ_q = ℓ_p = 0.5
            # L = diag(0.5, 0.5) → a=log(0.5), b=0, c=log(0.5)
            self.chol = np.zeros((self.M, 3))
            self.chol[:, 0] = np.log(0.5)   # a = log(L_11)
            self.chol[:, 1] = 0.0            # b = L_21
            self.chol[:, 2] = np.log(0.5)   # c = log(L_22)
        else:
            self.chol = np.asarray(chol_params, dtype=float)

        self.log_sn = float(log_sn)

    def _get_L(self, i):
        """Return the 2×2 lower-triangular Cholesky factor L_i."""
        a, b, c = self.chol[i]
        return np.array([[np.exp(a), 0.0],
                         [b,         np.exp(c)]])

    def _get_Sinv(self, i):
        """Return Σ_i^{-1} = L_i^{-T} L_i^{-1}."""
        L = self._get_L(i)
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv

    def get_ellipse_params(self, i):
        """Return (center, width, height, angle_deg) for matplotlib Ellipse."""
        L = self._get_L(i)
        Sigma = L @ L.T
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return self.centers[i], 2*np.sqrt(eigvals[0]), 2*np.sqrt(eigvals[1]), angle

    @property
    def sigma_n(self): return np.exp(self.log_sn)
    @property
    def n_params(self): return 6 * self.M + 1

    def get_params(self):
        p = []
        for i in range(self.M):
            p += [self.log_lam[i],
                  self.chol[i, 0], self.chol[i, 1], self.chol[i, 2],
                  self.centers[i, 0], self.centers[i, 1]]
        p.append(self.log_sn)
        return np.array(p)

    def set_params(self, p):
        idx = 0
        for i in range(self.M):
            self.log_lam[i]    = p[idx]; idx += 1
            self.chol[i, 0]    = p[idx]; idx += 1
            self.chol[i, 1]    = p[idx]; idx += 1
            self.chol[i, 2]    = p[idx]; idx += 1
            self.centers[i, 0] = p[idx]; idx += 1
            self.centers[i, 1] = p[idx]; idx += 1
        self.log_sn = p[idx]

    def gram(self, Z1, Z2):
        N1, N2 = len(Z1), len(Z2)
        K = np.zeros((N1, N2))
        for i in range(self.M):
            lam = np.exp(self.log_lam[i])
            Sinv = self._get_Sinv(i)
            c = self.centers[i]
            d1 = Z1 - c                              # (N1, 2)
            d2 = Z2 - c                              # (N2, 2)
            # u^T Σ^{-1} u  for each row
            q1 = np.sum(d1 @ Sinv * d1, axis=1)      # (N1,)
            q2 = np.sum(d2 @ Sinv * d2, axis=1)      # (N2,)
            phi1 = np.exp(-0.5 * q1)
            phi2 = np.exp(-0.5 * q2)
            K += lam * np.outer(phi1, phi2)
        return K

    def __repr__(self):
        a = np.exp(self.log_lam)
        return f"NKN(M={self.M}, σn={self.sigma_n:.4f}, amp=[{a.min():.3f}..{a.max():.3f}], 6M+1={self.n_params}p)"


def _sq_dist(X1, X2):
    return (np.sum(X1**2, axis=1, keepdims=True)
            - 2.0 * X1 @ X2.T + np.sum(X2**2, axis=1))