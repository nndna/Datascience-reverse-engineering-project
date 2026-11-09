"""Tests how close a simulated recording is to the original one.

The simulation uses a new random spike train and new noise, so the two traces will never
match point by point. Instead, both files are fitted from scratch with the same pipeline
and every part of the model (spikes, drift, noise) is compared with a statistical test.
Each test asks: "is the difference between the two files bigger than random chance
would produce?" A p-value above 0.05 means no detectable difference.

Usage:
    python test_simulation.py                                  (default file names)
    python test_simulation.py spike_data.npz simulated_spike_data.npz --plot comparison.png
    python test_simulation.py --calibrate 100                  (checks the tests themselves)
    pytest test_simulation.py                                  (same tests, pytest style)
"""
import argparse
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.signal import welch

import spike_model as sm

ALPHA = 0.05


@dataclass
class Check:
    part: str
    name: str
    original: float
    simulated: float
    method: str
    p: float


def _z_test(a, se_a, b, se_b):
    z = (a - b) / np.hypot(se_a, se_b)
    return 2 * stats.norm.sf(abs(z))


def check_structure(path_a, path_b):
    """Exact checks: same keys, shapes, dtypes and the same time axis."""
    a, b = np.load(path_a), np.load(path_b)
    problems = []
    if sorted(a.files) != sorted(b.files):
        problems.append(f"keys differ: {sorted(a.files)} vs {sorted(b.files)}")
    else:
        for k in a.files:
            if a[k].shape != b[k].shape:
                problems.append(f"{k}: shape {a[k].shape} vs {b[k].shape}")
            if a[k].dtype != b[k].dtype:
                problems.append(f"{k}: dtype {a[k].dtype} vs {b[k].dtype}")
        if "t" in a.files and a["t"].shape == b["t"].shape and not np.array_equal(a["t"], b["t"]):
            problems.append("time axis t differs")
    return problems


def statistical_checks(fit_a, fit_b):
    """Compare two fitted recordings. Returns a list of Check objects."""
    pa, pb = fit_a["params"], fit_b["params"]
    sa, sb = fit_a["se"], fit_b["se"]
    na, nb = len(fit_a["onsets"]), len(fit_b["onsets"])
    checks = []

    # spikes
    p_rate = stats.binomtest(na, na + nb, fit_a["T"] / (fit_a["T"] + fit_b["T"])).pvalue
    checks.append(Check("spikes", "spike rate (Hz)", pa["rate"], pb["rate"],
                        "conditional test for two Poisson counts", p_rate))
    isi_a, isi_b = np.diff(fit_a["onsets"]) * fit_a["dt"], np.diff(fit_b["onsets"]) * fit_b["dt"]
    checks.append(Check("spikes", "ISI distribution (median ms)", np.median(isi_a) * 1000,
                        np.median(isi_b) * 1000, "two-sample KS", stats.ks_2samp(isi_a, isi_b).pvalue))
    checks.append(Check("spikes", "spike height A", pa["A"], pb["A"], "Welch t-test on per-spike heights",
                        stats.ttest_ind(fit_a["A_k"], fit_b["A_k"], equal_var=False).pvalue))
    checks.append(Check("spikes", "spike height spread (sd)", fit_a["A_k"].std(ddof=1),
                        fit_b["A_k"].std(ddof=1), "Levene test", stats.levene(fit_a["A_k"], fit_b["A_k"]).pvalue))
    checks.append(Check("spikes", "decay time tau (ms)", pa["tau"] * 1000, pb["tau"] * 1000,
                        "z-test on fitted values", _z_test(pa["tau"], sa["tau"], pb["tau"], sb["tau"])))

    # drift
    for k, label in [("D", "drift amplitude D"), ("f", "drift frequency f (Hz)"), ("phi", "drift phase phi (rad)")]:
        checks.append(Check("drift", label, pa[k], pb[k], "z-test on fitted values",
                            _z_test(pa[k], sa[k], pb[k], sb[k])))

    # noise
    ra, rb = fit_a["resid"], fit_b["resid"]
    F = ra.var(ddof=1) / rb.var(ddof=1)
    p_F = 2 * min(stats.f.sf(F, len(ra) - 1, len(rb) - 1), stats.f.cdf(F, len(ra) - 1, len(rb) - 1))
    checks.append(Check("noise", "noise level sigma", pa["sigma"], pb["sigma"], "F-test on variances", p_F))
    checks.append(Check("noise", "noise distribution (kurtosis)", stats.kurtosis(ra), stats.kurtosis(rb),
                        "two-sample KS on residuals", stats.ks_2samp(ra, rb).pvalue))
    # whiteness: compare autocorrelations at lags 1..20. Each sample autocorrelation of white
    # noise has variance about 1/n, so the scaled sum of squared differences is chi-square(20).
    lags = range(1, 21)
    ac_a = np.array([np.corrcoef(ra[:-k], ra[k:])[0, 1] for k in lags])
    ac_b = np.array([np.corrcoef(rb[:-k], rb[k:])[0, 1] for k in lags])
    Q = ((ac_a - ac_b) ** 2 / (1 / len(ra) + 1 / len(rb))).sum()
    checks.append(Check("noise", "noise whiteness (lag-1 autocorr)", ac_a[0], ac_b[0],
                        "chi-square on autocorr, lags 1-20", stats.chi2.sf(Q, len(ac_a))))
    return checks


def holm_pass(checks, alpha=ALPHA):
    """Overall verdict with the Holm correction, since many tests are run at once."""
    ps = np.sort([c.p for c in checks])
    m = len(ps)
    return all(p > alpha / (m - i) for i, p in enumerate(ps))


def similarity_metrics(t, xa, xb, fit_a, fit_b):
    """Descriptive closeness scores (no pass/fail), useful for seeing how close at a glance."""
    out = {}
    out["signal mean"] = (xa.mean(), xb.mean())
    out["signal sd"] = (xa.std(), xb.std())

    def avg_waveform(x, fit):
        base = sm.drift(t, *(fit["params"][k] for k in ["D", "f", "phi"]))
        w = [x[s:s + 15] - base[s:s + 15] for s in fit["onsets"] if s + 15 <= len(x)]
        return np.mean(w, axis=0)

    wa, wb = avg_waveform(xa, fit_a), avg_waveform(xb, fit_b)
    out["avg spike waveform correlation"] = (np.corrcoef(wa, wb)[0, 1], None)

    def slow(x, fit):
        y = x - sm.spike_train(t, fit["onsets"], fit["params"]["A"], fit["params"]["tau"])
        return np.convolve(y, np.ones(501) / 501, "valid")

    out["drift curve correlation (0.5 s smoothing)"] = (np.corrcoef(slow(xa, fit_a), slow(xb, fit_b))[0, 1], None)

    fs = 1 / (t[1] - t[0])
    f, Pa = welch(xa, fs=fs, nperseg=1024)
    _, Pb = welch(xb, fs=fs, nperseg=1024)
    band = (f >= 1) & (f <= fs / 2)
    out["power spectrum mean |log10 ratio|"] = (np.mean(np.abs(np.log10(Pa[band] / Pb[band]))), None)

    bins = np.linspace(min(xa.min(), xb.min()), max(xa.max(), xb.max()), 81)
    ha, _ = np.histogram(xa, bins)
    hb, _ = np.histogram(xb, bins)
    out["amplitude histogram overlap"] = (np.minimum(ha, hb).sum() / len(xa), None)
    return out


def compare(path_orig, path_sim, fit_orig=None):
    t, xa = sm.load(path_orig)
    t_b, xb = sm.load(path_sim)
    fit_a = fit_orig if fit_orig is not None else sm.fit(t, xa)
    fit_b = sm.fit(t_b, xb)
    return dict(structure=check_structure(path_orig, path_sim),
                checks=statistical_checks(fit_a, fit_b),
                metrics=similarity_metrics(t, xa, xb, fit_a, fit_b),
                fits=(fit_a, fit_b), data=(t, xa, xb))


def print_report(r):
    print("\n1) FILE STRUCTURE")
    if r["structure"]:
        for p in r["structure"]:
            print("   FAIL:", p)
    else:
        print("   PASS: same keys, shapes, dtypes and time axis")

    print("\n2) STATISTICAL TESTS  (PASS = no detectable difference, p > 0.05)")
    print(f"   {'part':7s} {'quantity':34s} {'original':>10s} {'simulated':>10s} {'p-value':>8s}  result  test")
    for c in r["checks"]:
        res = "PASS" if c.p > ALPHA else "FAIL"
        print(f"   {c.part:7s} {c.name:34s} {c.original:10.4f} {c.simulated:10.4f} {c.p:8.3f}  {res:6s}  {c.method}")
    n_pass = sum(c.p > ALPHA for c in r["checks"])
    print(f"\n   {n_pass}/{len(r['checks'])} tests pass individually.")
    print("   Overall (Holm correction for running many tests):",
          "PASS" if holm_pass(r["checks"]) and not r["structure"] else "FAIL")

    print("\n3) SIMILARITY METRICS  (descriptive, closer to 1 or to matching values is better)")
    for k, (a, b) in r["metrics"].items():
        if b is None:
            print(f"   {k:45s} {a:.4f}")
        else:
            print(f"   {k:45s} original {a:.4f}   simulated {b:.4f}")


def plot_comparison(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, xa, xb = r["data"]
    fit_a, fit_b = r["fits"]
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))

    ax[0, 0].plot(t, xa, label="original")
    ax[0, 0].plot(t, xb, alpha=0.6, label="simulated")
    ax[0, 0].set_title("Traces")
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].legend()

    for fit, x, lab in [(fit_a, xa, "original"), (fit_b, xb, "simulated")]:
        base = sm.drift(t, *(fit["params"][k] for k in ["D", "f", "phi"]))
        w = np.mean([x[s:s + 15] - base[s:s + 15] for s in fit["onsets"] if s + 15 <= len(x)], axis=0)
        ax[0, 1].plot(np.arange(15), w, marker="o", label=lab)
    ax[0, 1].set_title("Average spike waveform")
    ax[0, 1].set_xlabel("ms after onset")
    ax[0, 1].legend()

    for fit, lab in [(fit_a, "original"), (fit_b, "simulated")]:
        isi = np.sort(np.diff(fit["onsets"])) * fit["dt"] * 1000
        ax[1, 0].step(isi, np.arange(1, len(isi) + 1) / len(isi), where="post", label=lab)
    ax[1, 0].set_title("ISI distribution (CDF)")
    ax[1, 0].set_xlabel("ISI (ms)")
    ax[1, 0].legend()

    bins = np.linspace(-1.5, 1.5, 61)
    ax[1, 1].hist(fit_a["resid"], bins, density=True, histtype="step", label="original")
    ax[1, 1].hist(fit_b["resid"], bins, density=True, histtype="step", label="simulated")
    ax[1, 1].set_title("Noise (residual) distribution")
    ax[1, 1].legend()

    fs = 1 / (t[1] - t[0])
    for x, lab in [(xa, "original"), (xb, "simulated")]:
        f, P = welch(x, fs=fs, nperseg=1024)
        ax[2, 0].semilogy(f, P, label=lab)
    ax[2, 0].set_title("Power spectrum")
    ax[2, 0].set_xlabel("frequency (Hz)")
    ax[2, 0].legend()

    names = [c.name for c in r["checks"]]
    ax[2, 1].barh(names, [c.p for c in r["checks"]])
    ax[2, 1].axvline(ALPHA, linestyle="--")
    ax[2, 1].set_xlim(0, 1)
    ax[2, 1].invert_yaxis()
    ax[2, 1].set_title("p-value per test (right of dashed line = pass)")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def calibrate(path_orig, n=100):
    """Compare the original against n fresh simulations. If the model is right, each test
    should pass at least about 95% of the time (usually more, because the simulations are
    centred on the original's own fitted values)."""
    t, xa = sm.load(path_orig)
    fit_a = sm.fit(t, xa)
    passes, overall = [], []
    for seed in range(1000, 1000 + n):
        xb, _ = sm.simulate(t, fit_a["params"], seed=seed)
        fit_b = sm.fit(t, xb)
        checks = statistical_checks(fit_a, fit_b)
        passes.append([c.p > ALPHA for c in checks])
        overall.append(holm_pass(checks))
    passes = np.array(passes)
    print(f"\nCALIBRATION: original vs {n} fresh simulations")
    for name, rate in zip([c.name for c in checks], passes.mean(0)):
        print(f"   {name:34s} passes {rate*100:5.1f}% of the time")
    print(f"   overall Holm verdict passes {np.mean(overall)*100:.1f}% of the time")


# ---------- pytest entry points (use the default file names in this folder) ----------
ORIG, SIM = "spike_data.npz", "simulated_spike_data.npz"


def _result():
    if not hasattr(_result, "cache"):
        _result.cache = compare(ORIG, SIM)
    return _result.cache


def test_file_structure():
    assert check_structure(ORIG, SIM) == []


def test_statistics_match():
    failed = [f"{c.name} (p={c.p:.3f})" for c in _result()["checks"] if c.p <= ALPHA]
    assert holm_pass(_result()["checks"]), "differences detected: " + ", ".join(failed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original", nargs="?", default=ORIG)
    ap.add_argument("simulated", nargs="?", default=SIM)
    ap.add_argument("--plot", help="save a comparison figure to this path")
    ap.add_argument("--calibrate", type=int, metavar="N", help="also run N fresh simulations to check the tests")
    args = ap.parse_args()

    r = compare(args.original, args.simulated)
    print_report(r)
    if args.plot:
        plot_comparison(r, args.plot)
        print(f"\nSaved figure to {args.plot}")
    if args.calibrate:
        calibrate(args.original, args.calibrate)


if __name__ == "__main__":
    main()
