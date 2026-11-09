# Reverse engineering `spike_data.npz`

This project figures out the process that generated the spike recording in `spike_data.npz`, turns it into a model, generates a new recording from that model, and tests how close the new recording is to the original.

## The data

`spike_data.npz` has two arrays:

| Key | Shape | Type | Contents |
|---|---|---|---|
| `t` | (10000,) | float64 | time in seconds, 0 to 9.999 s in 1 ms steps (1 kHz) |
| `raw_data` | (10000,) | float64 | the recorded signal |

## The model

The signal is the sum of three parts: a slow drift, spikes, and noise.

$$x(t) = \underbrace{D\,\sin(2\pi f t + \varphi)}_{\text{drift}} + \underbrace{\sum_{k} A\, e^{-(t - t_k)/\tau}\, H(t - t_k)}_{\text{spikes}} + \underbrace{\varepsilon(t)}_{\text{noise}}$$

* **Drift:** a slow sine wave with amplitude $D$, frequency $f$ and phase $\varphi$.
* **Spikes:** each spike starts at time $t_k$, jumps up by $A$ in a single sample, and decays exponentially with time constant $\tau$. $H(t - t_k)$ is the Heaviside step function, so a spike is zero before it starts. The spike times $t_k$ come from a homogeneous Poisson process with rate $\lambda$.
* **Noise:** $\varepsilon(t) \sim \mathcal{N}(0, \sigma^2)$, independent at every sample (white Gaussian noise).

### Fitted parameters

| Parameter | Meaning | Value |
|---|---|---|
| $D$ | drift amplitude | 0.6466 ± 0.0042 |
| $f$ | drift frequency | 0.0908 ± 0.0004 Hz |
| $\varphi$ | drift phase | 0.2912 ± 0.0143 rad |
| $A$ | spike height | 5.0177 ± 0.0397 |
| $\tau$ | spike decay time constant | 2.989 ± 0.039 ms |
| $\sigma$ | noise standard deviation | 0.2984 |
| $\lambda$ | spike rate | 4.2 Hz (42 spikes, 95% CI 3.0 to 5.5 Hz) |

## Files

| File | What it is |
|---|---|
| `spike_data.npz` | the original recording (input) |
| `spike_model.ipynb` | step-by-step notebook showing how the model was found, with plots and explanations |
| `spike_model.py` | the model: spike detection, fitting, simulation; running it writes the results file |
| `simulated_spike_data.npz` | results file: a new 10 s recording generated from the model, same structure as the original |
| `test_simulation.py` | testing module that measures how close the simulation is to the original |
| `comparison.png` | comparison figure produced by the testing module |

## Requirements

Python 3 with `numpy`, `scipy` and `matplotlib`. `pytest` is optional (to run the tests pytest style) and `jupyter` is needed to open the notebook.

```bash
pip install numpy scipy matplotlib pytest jupyter
```

Keep all the files in the same folder.

## How to run

**Generate the results file** (fits the model to the original, then simulates a new recording):

```bash
python spike_model.py spike_data.npz simulated_spike_data.npz --seed 0
```

Changing `--seed` gives a different random recording from the same model.

**Test how close the simulation is to the original:**

```bash
python test_simulation.py spike_data.npz simulated_spike_data.npz --plot comparison.png
```

or, with the default file names:

```bash
pytest test_simulation.py
```

**Check the tests themselves** (compares the original against N fresh simulations):

```bash
python test_simulation.py --calibrate 200
```

## How the model was found

The full reasoning is in `spike_model.ipynb`. In short, each part of the equation was added to explain something visible in the data, and after each step what had been explained was subtracted to see what was left.

1. **Find the spikes.** Every spike starts with a jump of about 5 in one sample, while noise only changes the signal by about 0.4 between samples. A threshold of 2.5 on the sample-to-sample difference finds all 42 spikes.
2. **Learn the spike shape.** Least squares was used to estimate one shared spike shape without assuming its form. It shrinks to about 0.72 of its previous value every millisecond, which means it is an exponential decay with $\tau = -1/\ln(0.72) \approx 3$ ms.
3. **Find the drift.** After removing the spikes, neighboring samples were still correlated (about 0.72), so something slow was left. It rises and falls once over 10 s. Compared by BIC against a straight line and polynomials, a sine wave fit best.
4. **Fit everything together.** All parameters were fit at once with nonlinear least squares so the estimates don't bias each other. Per-spike heights did not vary more than noise alone explains (p = 0.55), so one shared height $A$ is enough.
5. **Check the noise.** After subtracting spikes and drift, the leftover is Gaussian and uncorrelated across lags, so it is white noise.
6. **Check the spike timing.** The gaps between spikes follow an exponential distribution (p = 0.72), spikes are spread evenly over time (p = 0.45), and the drift has no effect on when spikes happen (p = 0.37). That points to a homogeneous Poisson process.
7. **Test the method on fake data.** On 50 recordings simulated with known parameters, the same pipeline recovered every parameter without bias and found the exact spike count in all 50.

## Testing module

The simulation has its own random spike times and noise, so it can never match the original point by point. `test_simulation.py` fits both files from scratch with the same pipeline and then compares them in three ways.

1. **File structure:** same keys, shapes, dtypes and time axis.
2. **Statistical tests:** 11 tests, each asking whether a difference is larger than chance would produce. A p-value above 0.05 counts as a pass. The overall verdict uses the Holm correction, since many tests are run at once.
3. **Similarity metrics:** descriptive scores such as how well the average spike shapes and drift curves correlate and how much the value histograms overlap.

### Results for `simulated_spike_data.npz` (seed 0)

| Part | Quantity | Original | Simulated | p-value | Result |
|---|---|---|---|---|---|
| spikes | spike rate (Hz) | 4.20 | 3.90 | 0.824 | PASS |
| spikes | ISI distribution (median ms) | 169.0 | 168.5 | 0.992 | PASS |
| spikes | spike height A | 5.018 | 5.055 | 0.442 | PASS |
| spikes | spike height spread (sd) | 0.204 | 0.225 | 0.645 | PASS |
| spikes | decay time τ (ms) | 2.989 | 3.018 | 0.612 | PASS |
| drift | amplitude D | 0.647 | 0.645 | 0.773 | PASS |
| drift | frequency f (Hz) | 0.0908 | 0.0909 | 0.863 | PASS |
| drift | phase φ (rad) | 0.291 | 0.283 | 0.701 | PASS |
| noise | level σ | 0.298 | 0.296 | 0.442 | PASS |
| noise | distribution | | | 0.688 | PASS |
| noise | whiteness (lags 1 to 20) | | | 0.159 | PASS |

All 11 tests pass and the overall verdict is PASS. The file structure check also passes. Similarity scores: average spike shape correlation 0.998, drift curve correlation 0.999, value histogram overlap 97%.

### Are the tests trustworthy?

* **When the simulation is correct:** across 200 fresh simulations, the overall verdict passed 98.5% of the time, so the tests rarely flag a correct model by mistake.
* **When the simulation is wrong:** each setting was deliberately changed and tried 40 times. A decay time of 4 ms instead of 3 ms, a spike height of 5.5, a noise level of 0.33, and a wrong drift amplitude or frequency were each caught 40 out of 40 times. A doubled spike rate (8 Hz) was caught 29 out of 40 times, because 10 s of data has too few spikes to pin the rate down tightly.

## Limitations

* **Drift shape:** 10 s of data only shows about one cycle of the drift, so the exact shape is less certain than the other parts. A sine wave fits best of the shapes tested.
* **Spike rate:** 42 spikes only narrow the rate down to about 3.0 to 5.5 Hz.
* **Refractory period:** the shortest gap between spikes is 11 ms. That is consistent with no refractory period, but a short one can't be ruled out.
* **Timing resolution:** at 1 kHz, spike onsets are only known to the nearest millisecond.
