# MultiHiggsFitting
A reusable MadGraph pipeline for scanning multi-Higgs production over anomalous
couplings and fitting cross sections or binned distributions.

The initial configs reproduce the `/mnt/ssd2/Projects/4H` `loop_sm_c3d4`
workflow for:

- `gg -> h h h` in [configs/gg_hhh_c3d4.toml](configs/gg_hhh_c3d4.toml)
- `gg -> h h h h` in [configs/gg_hhhh_c3d4.toml](configs/gg_hhhh_c3d4.toml)

## Setup

From this repository:

```bash
python3 -m pip install -e .
```

For ad hoc use without installing, run commands as:

```bash
python3 -m multihiggs grid configs/gg_hhh_c3d4.toml
```

The code expects `numpy` and `tomli` on Python 3.10. They are declared in
`pyproject.toml`.

## Basic Workflow

1. Inspect the scan points:

```bash
multihiggs grid configs/gg_hhh_c3d4.toml
```

This writes `outputs/gg_hhh/scan_points.csv`.

2. Generate an MG5 process card:

```bash
multihiggs generate-process configs/gg_hhh_c3d4.toml
```

By default this only writes the card. To actually run MG5:

```bash
multihiggs generate-process configs/gg_hhh_c3d4.toml --run
```

Use `--force-output` if you want the MG5 `output ... -f` behavior.

3. Write the madevent scan command file:

```bash
multihiggs scan configs/gg_hhh_c3d4.toml
```

This skips completed runs by checking for
`Events/<run_name>/unweighted_events.lhe.gz`. To run madevent:

```bash
multihiggs scan configs/gg_hhh_c3d4.toml --run
```

Useful development options:

```bash
multihiggs scan configs/gg_hhh_c3d4.toml --max-runs 3
multihiggs scan configs/gg_hhh_c3d4.toml --force
```

4. Collect completed cross sections:

```bash
multihiggs collect configs/gg_hhh_c3d4.toml --run-number 2
```

This writes `outputs/gg_hhh/xsecs.csv`.

5. Fit the cross section:

```bash
multihiggs fit configs/gg_hhh_c3d4.toml
```

This writes `outputs/gg_hhh/fit.json`, including Chebyshev coefficients,
the covariance matrix, fit diagnostics, and monomial coefficients normalized
to the SM point.

## Adapting To A New Process Or Model

Copy one of the config files and edit:

- `[process]`: `name`, `mg5_path`, `model`, `generate`, and `output`
- `[[couplings]]`: one entry per MadGraph parameter to scan
- `[scan]`: run number, collider energy, events per point, and integration options
- `[fit]`: Chebyshev terms to fit

For the 4H-style `c3,d4` scan, the fitted variables are physical kappas:

```toml
name = "c3"
parameter = "c3"
fit_name = "k3"
range = [0.0, 2.0]
fit_offset = 1.0
```

This means the grid is built in `k3 = c3 + 1`, while MadGraph receives `c3`.
For a parameter scanned directly, set `fit_offset = 0.0`.

## Distribution Fits

Configs can define `[[observables]]` for simple LHE histograms. Example:

```toml
[[observables]]
name = "hh_mass"
kind = "invariant_mass"
pdg_ids = [25, 25]
which = "all"
bins = [200.0, 300.0, 400.0, 500.0, 700.0, 1000.0]
```

Build histograms:

```bash
multihiggs hist configs/gg_hhh_c3d4.toml
```

Fit every histogram bin with the same polynomial basis:

```bash
multihiggs fit configs/gg_hhh_c3d4.toml \
  --input outputs/gg_hhh/histograms.csv \
  --output outputs/gg_hhh/hist_fit.json
```

For per-object observables such as `h_pt` with `which = "all"`, the histogram
integral is multiplicity times the event cross section.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
