"""
gp.py — Scalar GP regression for smoothing the Hessian field.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def nll(kernel, X, y, jitter=1e-6):
    N = len(X)
    K = kernel.gram(X, X) + (kernel.sigma_n**2 + jitter) * np.eye(N)
    try:
        L, lower = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = cho_solve((L, lower), y)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * y @ alpha + 0.5 * log_det + 0.5 * N * np.log(2 * np.pi)


def fit(kernel, X, y, max_iter=200, verbose=True):
    p0 = kernel.get_params()
    calls = [0]
    def obj(params):
        kernel.set_params(params)
        v = nll(kernel, X, y)
        calls[0] += 1
        if verbose and calls[0] % 100 == 0:
            print(f"      iter {calls[0]:5d}  NLL={v:.2f}")
        return v
    res = minimize(obj, p0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-9})
    kernel.set_params(res.x)
    if verbose:
        print(f"      Done: NLL={res.fun:.2f} ({res.success})")
    return res


def predict(kernel, X_train, y_train, X_test, jitter=1e-6):
    N = len(X_train)
    K = kernel.gram(X_train, X_train) + (kernel.sigma_n**2 + jitter) * np.eye(N)
    L, lower = cho_factor(K, lower=True)
    alpha = cho_solve((L, lower), y_train)
    K_star = kernel.gram(X_test, X_train)
    mu = K_star @ alpha
    V = cho_solve((L, lower), K_star.T)
    var = np.diag(kernel.gram(X_test, X_test)) - np.sum(K_star.T * V, axis=0)
    return mu, np.clip(var, 0, None)


# ─── Heteroscedastic GP: input-dependent noise ──────────────────────────────

def fit_heteroscedastic(kernel, X, y, noise_var_per_point, max_iter=200,
                        verbose=True):
    """
    Fit kernel hyperparams with a KNOWN, position-dependent noise diagonal.

    noise_var_per_point : (N,) — the noise variance σ²(z_i) at each point.
    Only the kernel's own params (not σ_n) are optimised; the noise is fixed.
    """
    # We'll still use kernel.sigma_n as a small floor
    p0 = kernel.get_params()
    calls = [0]
    N = len(X)
    Lambda = np.diag(noise_var_per_point)

    def obj(params):
        kernel.set_params(params)
        K = kernel.gram(X, X) + Lambda + 1e-6 * np.eye(N)
        try:
            L_chol, lower = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve((L_chol, lower), y)
        log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
        val = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * N * np.log(2*np.pi)
        calls[0] += 1
        if verbose and calls[0] % 100 == 0:
            print(f"      iter {calls[0]:5d}  NLL={val:.2f}")
        return val

    res = minimize(obj, p0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-9})
    kernel.set_params(res.x)
    if verbose:
        print(f"      Done: NLL={res.fun:.2f} ({res.success})")
    return res


def predict_heteroscedastic(kernel, X_train, y_train, X_test,
                            noise_var_per_point, jitter=1e-6):
    """
    Predict with known heteroscedastic noise. Returns:
      mu    — posterior mean
      var_f — posterior variance of the FUNCTION (epistemic)
      var_y — predictive variance of a new obs = var_f + σ²(z*)

    For test points, we estimate σ²(z*) by nearest-neighbour interpolation
    from the training noise field.
    """
    N = len(X_train)
    Lambda = np.diag(noise_var_per_point)
    K = kernel.gram(X_train, X_train) + Lambda + jitter * np.eye(N)
    L, lower = cho_factor(K, lower=True)
    alpha = cho_solve((L, lower), y_train)

    K_star = kernel.gram(X_test, X_train)
    mu = K_star @ alpha

    V = cho_solve((L, lower), K_star.T)
    var_f = np.diag(kernel.gram(X_test, X_test)) - np.sum(K_star.T * V, axis=0)
    var_f = np.clip(var_f, 0, None)

    # Interpolate noise to test points (nearest neighbour)
    from scipy.spatial import cKDTree
    tree = cKDTree(X_train)
    # For repeated training inputs, average the noise at each unique location
    _, idx = tree.query(X_test)
    noise_at_test = noise_var_per_point[idx]

    var_y = var_f + noise_at_test
    return mu, var_f, var_y
