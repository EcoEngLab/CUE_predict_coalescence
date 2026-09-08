# Carbon-Use Efficiency Predicts Microbial Community Coalescence

Simulation and analysis code for the study **“Carbon-Use Efficiency Predicts the Outcome of Microbial Community Coalescence.”** The repository uses a microbial consumer–resource model (MiCRM) to assemble two communities, coalesce them, and test how species- and community-level carbon-use efficiency (CUE) relates to survival, abundance, dominance, feasibility, and local stability.

## Overview

Each simulation draws species and resources from a larger pool, integrates the two parent communities to equilibrium, and then initializes a coalesced community from their final abundances. The scripts cover three related experiments:

- Baseline coalescence, including mechanistic CUE–abundance predictions and stability analysis.
- Rare-species invasion at different invader dilution rates.
- Coalescence across different levels of resource overlap.

For species \(i\) and resource \(a\), the model evolves consumer abundance \(C_i\) and resource concentration \(R_a\) as

$$
\frac{dC_i}{dt}
= C_i\left(\sum_a u_{ia}\eta_{ia}R_a-m_i\right),
$$

$$
\frac{dR_a}{dt}
= \rho_a-\omega_aR_a
-\sum_i C_i u_{ia}R_a
+\sum_{i,\beta}C_iR_\beta u_{i\beta}l_{i\beta a},
$$

where \(u\) is the uptake matrix, \(l\) is the leakage tensor, and \(m\) is the maintenance cost. Retained carbon is

$$
\eta_{ia}=1-\sum_{\beta}l_{ia\beta}.
$$

Species CUE is evaluated in a reference resource environment as net growth flux divided by total uptake flux. Community CUE is the abundance-weighted mean of its species’ CUE values.

## Repository contents

| File | Purpose | Main output |
| --- | --- | --- |
| `main.py` | Baseline community assembly and coalescence; CUE–abundance theory; monoculture CUE assays; feasibility and Jacobian stability metrics | `coal.csv`, `coal_summary.csv`, `cue_abundance_theory_params.csv`, `rmax_cue.csv` |
| `dilution.py` | Rare-species invasion with the second community introduced at multiple dilution rates | `rare.csv` |
| `resource_overlap.py` | Coalescence at 25%, 50%, and 75% resource overlap | `coal_resource.csv` |
| `plot.py` | Analysis and figure generation from the simulation CSV files | PNG/PDF files under `figures/`, plus interactive figures |

Generated data and figures are written beside the scripts and are not required to be present before the simulations are run.

## Requirements

- Python 3.9 or newer
- NumPy
- pandas
- SciPy
- Matplotlib
- seaborn

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib seaborn
```

## Running the simulations

Run commands from the repository root.

### Baseline coalescence

```bash
python main.py
```

The default run evaluates 50 random realizations with up to four worker processes. It also performs one monoculture assay for every Community 1 species in every realization. This is computationally intensive; progress is printed as seeds complete and every 30 seconds while waiting.

For a quick test, edit the parameter block near the top of `main.py`, for example:

```python
N_SIMULATIONS = 2
N_WORKERS = 2
COMPUTE_MEASURABLE_CUE = False
```

Setting `COMPUTE_MEASURABLE_CUE = False` skips the additional monoculture ODE solves. `rmax_cue.csv` is still created, but its assay-derived columns will be empty.

### Full experiment and plotting workflow

To generate every dataset expected by `plot.py`:

```bash
python main.py
python dilution.py
python resource_overlap.py
python plot.py
```

`dilution.py` defaults to 100 seeds at dilution rates 0.01 and 0.1. `resource_overlap.py` defaults to 50 seeds at overlap ratios 0.25, 0.5, and 0.75. Both use all CPU cores reported by Python. Adjust their parameter blocks before running on a shared machine or when testing locally.

Some plots are saved as both PNG and PDF under `figures/`, including the `cue_stability/`, `monoculture_figures/`, and `existing_plots/` subdirectories; others are displayed through Matplotlib. On a headless system, set an appropriate Matplotlib backend.

## Main outputs

### `coal.csv`

Species-level results for Communities 1 and 2 and the coalesced Community 3. The table includes:

- species and community CUE;
- final and theory-predicted abundance;
- competition, facilitation, depletion, and uptake-variance metrics;
- mechanistic CUE-threshold quantities;
- integration and equilibrium diagnostics;
- feasibility-proxy and local-stability results;
- measurable monoculture CUE fields for Community 1.

### `coal_summary.csv`

Community-level averages of CUE, abundance, predicted abundance, competition, facilitation, depletion, and uptake variance.

### `cue_abundance_theory_params.csv`

Aggregated parameters for the mechanistic saturating CUE–abundance curve, including the estimated CUE threshold, response scale, maximum abundance, and log-scale \(R^2\).

### `rmax_cue.csv`

Community 1 monoculture assays comparing theoretical/intrinsic CUE with growth-derived measurable CUE.

### `rare.csv` and `coal_resource.csv`

Outputs for the dilution and resource-overlap experiments used by the full plotting workflow.

## Configuration and reproducibility

Simulation settings are constants near the top of each script. The most useful parameters to change are:

- `BASE_SEED` and `N_SIMULATIONS` for reproducibility and replicate count;
- `N_POOL`, `M_POOL`, `N1`, `N2`, `M1`, and `M2` for pool and community sizes;
- `LEAKAGE_RATE`, `MAINTENANCE_COST`, `RHO_VALUE`, and `OMEGA_VALUE` for model physiology and resource dynamics;
- `T_SPAN` and `SURVIVAL_THRESHOLD` for integration and survival criteria;
- `DILUTION_RATES` in `dilution.py`;
- `OVERLAP_RATIOS` in `resource_overlap.py`.

Random numbers are generated with NumPy’s `default_rng` from a fixed base seed. The baseline workflow preserves seed order in its CSV output even though simulations finish in parallel. Existing output files with the same names are overwritten when a workflow completes.

The feasibility value reported by `main.py` is an SVD log-volume proxy per survivor dimension, not a probability. Stability is assessed from the largest real part of the eigenvalues of the full consumer–resource Jacobian at endpoints that meet the numerical equilibrium tolerance.

## Citation

If you use this code, please cite the associated paper:

> *Carbon-Use Efficiency Predicts the Outcome of Microbial Community Coalescence.*

Add the authors, journal, year, and DOI here when the final bibliographic record is available.

## License

This project is released under the [MIT License](LICENSE).
