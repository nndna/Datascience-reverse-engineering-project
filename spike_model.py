"""Generative model for spike_data.npz: drift + spikes + noise.

    x(t) = D sin(2 pi f t + phi)                         drift
         + sum_k A exp(-(t - t_k)/tau) H(t - t_k)        spikes (t_k from a Poisson process)
         + eps(t),  eps(t) ~ N(0, sigma^2)               white Gaussian noise

Running this file fits the model to the original recording and writes a simulated
recording with the same structure (keys "t" and "raw_data", same shape and dtype):

    python spike_model.py spike_data.npz simulated_spike_data.npz --seed 0
"""
import argparse

import numpy as np
from scipy import optimize

PARAM_NAMES = ["D", "f", "phi", "A", "tau"]


def load(path):
    d = np.load(path)
    return d["t"], d["raw_data"]


def save_results(path, t, x):
    # same keys, shapes and dtypes as the original file
    np.savez(path, t=np.asarray(t, dtype=np.float64), raw_data=np.asarray(x, dtype=np.float64))


def detect_spikes(x, jump=2.5):
    # a spike onset is a single-sample jump of about 5; noise only moves about 0.4 per sample.
    # A plain threshold on the jump (not a peak finder) also catches two spikes in back-to-back
    # samples, because the second one still adds a jump of about 3.6.
    dx = np.diff(x, prepend=x[0])
    return np.flatnonzero(dx > jump)


def drift(t, D, f, phi):
    return D * np.sin(2 * np.pi * f * t + phi)


def spike_train(t, onsets, A, tau):
    y = np.zeros(len(t))
    amps = np.broadcast_to(A, len(onsets))
    for s, a in zip(onsets, amps):
        y[s:] += a * np.exp(-(t[s:] - t[s]) / tau)
    return y


def model(p, t, onsets):
    D, f, phi, A, tau = p
    return drift(t, D, f, phi) + spike_train(t, onsets, A, tau)


def fit(t, x, onsets=None):
    """Fit the model to one recording. Returns a dict with estimates, standard errors,
    the residual (noise estimate) and the per-spike heights A_k."""
    if onsets is None:
        onsets = detect_spikes(x)
    p0 = [0.6, 0.1, 0.0, 5.0, 0.003]
    res = optimize.least_squares(lambda p: model(p, t, onsets) - x, p0,
                                 x_scale=[0.1, 0.01, 0.1, 1, 0.001])
    p = res.x
    resid = x - model(p, t, onsets)
    cov = np.linalg.inv(res.jac.T @ res.jac) * resid.var()
    se = np.sqrt(np.diag(cov))

    # per-spike heights A_k, with drift and tau held at their fitted values
    D, f, phi, A, tau = p
    K = np.zeros((len(t), len(onsets)))
    for j, s in enumerate(onsets):
        K[s:, j] = np.exp(-(t[s:] - t[s]) / tau)
    A_k, *_ = np.linalg.lstsq(K, x - drift(t, D, f, phi), rcond=None)

    dt = t[1] - t[0]
    T = len(t) * dt
    params = dict(zip(PARAM_NAMES, p))
    params["sigma"] = resid.std()
    params["rate"] = len(onsets) / T
    return dict(params=params, se=dict(zip(PARAM_NAMES, se)), resid=resid,
                onsets=onsets, A_k=A_k, T=T, dt=dt)


def simulate(t, params, seed=None):
    """Generate a new recording on the time grid t. Returns (x, onsets)."""
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    onsets = np.flatnonzero(rng.random(len(t)) < params["rate"] * dt)
    x = (drift(t, params["D"], params["f"], params["phi"])
         + spike_train(t, onsets, params["A"], params["tau"])
         + rng.normal(0, params["sigma"], len(t)))
    return x, onsets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default="spike_data.npz")
    ap.add_argument("out", nargs="?", default="simulated_spike_data.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t, x = load(args.data)
    est = fit(t, x)
    print("Fitted parameters:")
    for k, v in est["params"].items():
        print(f"  {k:6s} = {v:.6g}")

    x_sim, onsets = simulate(t, est["params"], seed=args.seed)
    save_results(args.out, t, x_sim)
    print(f"Wrote {args.out} ({len(onsets)} simulated spikes, seed {args.seed})")


if __name__ == "__main__":
    main()
