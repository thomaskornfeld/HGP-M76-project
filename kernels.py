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
    Per inducing point: log λ_i, log ℓ_iq, log ℓ_ip, p_iq, p_ip.
    """

    def __init__(self, centers, log_lam=None, log_ell=None, log_sn=-2.0):
        self.M = len(centers)
        self.centers = np.array(centers, dtype=float)
        self.log_lam = np.zeros(self.M) if log_lam is None else np.asarray(log_lam, dtype=float)
        self.log_ell = np.zeros((self.M, 2)) if log_ell is None else np.asarray(log_ell, dtype=float)
        self.log_sn = float(log_sn)

    @property
    def sigma_n(self): return np.exp(self.log_sn)
    @property
    def n_params(self): return 5 * self.M + 1
    def get_params(self):
        p = []
        for i in range(self.M):
            p += [self.log_lam[i], self.log_ell[i,0], self.log_ell[i,1],
                  self.centers[i,0], self.centers[i,1]]
        p.append(self.log_sn)
        return np.array(p)
    def set_params(self, p):
        idx = 0
        for i in range(self.M):
            self.log_lam[i]    = p[idx]; idx+=1
            self.log_ell[i, 0] = p[idx]; idx+=1
            self.log_ell[i, 1] = p[idx]; idx+=1
            self.centers[i, 0] = p[idx]; idx+=1
            self.centers[i, 1] = p[idx]; idx+=1
        self.log_sn = p[idx]

    def gram(self, Z1, Z2):
        N1, N2 = len(Z1), len(Z2)
        K = np.zeros((N1, N2))
        for i in range(self.M):
            lam = np.exp(self.log_lam[i])
            ell = np.exp(self.log_ell[i])
            c = self.centers[i]
            d1 = (Z1 - c) / ell
            d2 = (Z2 - c) / ell
            phi1 = np.exp(-0.5 * np.sum(d1**2, axis=1))
            phi2 = np.exp(-0.5 * np.sum(d2**2, axis=1))
            K += lam * np.outer(phi1, phi2)
        return K

    def __repr__(self):
        a = np.exp(self.log_lam)
        return f"NKN(M={self.M}, σn={self.sigma_n:.4f}, amp=[{a.min():.3f}..{a.max():.3f}])"


def _sq_dist(X1, X2):
    return (np.sum(X1**2, axis=1, keepdims=True)
            - 2.0 * X1 @ X2.T + np.sum(X2**2, axis=1))
