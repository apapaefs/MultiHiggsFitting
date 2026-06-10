# MultiHiggsFitting
A reusable MadGraph pipeline for scanning multi-Higgs production over anomalous
couplings and fitting cross sections or binned distributions.

The initial configs reproduce the `/mnt/ssd2/Projects/4H` `loop_sm_c3d4`
workflow for:

- `gg -> h h h` in [configs/gg_hhh_c3d4.toml](configs/gg_hhh_c3d4.toml)
- `gg -> h h h h` in [configs/gg_hhhh_c3d4.toml](configs/gg_hhhh_c3d4.toml)

There are also light validation configs with only `c3` scanned over
`[-20, 20]`:

- `gg -> h h [noborn=QCD]` in [configs/gg_hh_c3_validation.toml](configs/gg_hh_c3_validation.toml)
- `gg -> t t~ h h` in [configs/gg_tthh_c3_validation.toml](configs/gg_tthh_c3_validation.toml)

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

## MadGraph Setup

Install the basic build tools used by MG5/aMC and generated matrix elements:

```bash
sudo apt-get update
sudo apt-get install -y python3 gfortran g++ make wget tar gzip
```

Install or unpack MG5/aMC. For a local checkout, put the MG5 directory at the
repository root:

```bash
tar -xf LTS_MG5aMC_v3.5.15.tar
```

The example configs assume:

```toml
mg5_path = "MG5_aMC_v3_5_15"
```

If MG5 is somewhere else, edit `mg5_path` in the config you are using. The
original Linux workstation setup used
`/mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15`.

Install the UFO model used by the starter configs. The repository includes a
clean copy under `models/loop_sm_c3d4`; copy or symlink it into the MG5 model
directory. For the repo-local MG5 install:

```bash
ln -s ../../models/loop_sm_c3d4 MG5_aMC_v3_5_15/models/loop_sm_c3d4
```

The repository also vendors `heft_loop_sm_restricted5` from
`https://gitlab.com/apapaefs/multihiggs_loop_sm` at commit `99ba5ee90669`:

```bash
ln -s ../../models/heft_loop_sm_restricted5 MG5_aMC_v3_5_15/models/heft_loop_sm_restricted5
```

If you are reproducing the old 4H area and prefer the archived copy, it is also
available as `loop_sm_c3d4.tar.gz`:

```bash
tar -xzf /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/models/loop_sm_c3d4.tar.gz \
  -C /mnt/ssd2/Projects/4H/MG5_aMC_v3_5_15/models
```

Check that MG5 can import the model:

```bash
cd MG5_aMC_v3_5_15
./bin/mg5_aMC
```

Inside MG5:

```text
import model loop_sm_c3d4
display particles
quit
```

Loop-induced multi-Higgs runs need the loop libraries available at runtime.
The pipeline automatically adds common MG5 `HEPTools` and `COLLIER` library
paths under `mg5_path` to `LD_LIBRARY_PATH` when it launches MG5 or madevent.

The generated madevent scan cards explicitly disable common run-card cuts by
default. Each launch block sets:

```text
set dsqrt_shat 0.0
set ptheavy 0.0
set pt_min_pdg {}
set pt_max_pdg {}
set eta_min_pdg {}
set eta_max_pdg {}
set mxx_min_pdg {}
```

To intentionally use cuts, set `no_cuts = false` in `[scan]` and put the
desired `set ...` commands in `extra_set_commands`.

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

This skips completed runs only when
`Events/<run_name>/unweighted_events.lhe.gz` exists and has at least the
configured event minimum. By default, that minimum is `[scan].nevents`, which
defaults to `10000` if omitted. To run madevent:

```bash
multihiggs scan configs/gg_hhh_c3d4.toml --run
```

Useful development options:

```bash
multihiggs scan configs/gg_hhh_c3d4.toml --max-runs 3
multihiggs scan configs/gg_hhh_c3d4.toml --force
```

After generating a process directory, inspect the matrix-element coupling
support that the generated HELAS code implies:

```bash
multihiggs infer-terms configs/gg_tthh_c3_validation.toml
```

This prints the inferred amplitude support and the corresponding candidate
`[fit].terms` for the cross section. Treat the output as a generated
cross-check, not a proof: cancellations or model subtleties can still remove
or add practical fit requirements, and the command ends with an explicit
`WARNING: CHECK` reminder.

4. Collect completed cross sections:

```bash
multihiggs collect configs/gg_hhh_c3d4.toml --run-number 2
```

This writes `outputs/gg_hhh/xsecs.csv`. `collect` uses the same configured
event-count minimum as `scan`, so it will skip lower-stat run directories and
print a warning. To collect older one-event integration samples, override the
threshold:

```bash
multihiggs collect configs/gg_hhh_c3d4.toml --run-number 2 --min-events 1
```

5. Fit the cross section:

```bash
multihiggs fit configs/gg_hhh_c3d4.toml
```

This writes `outputs/gg_hhh/fit.json`, including Chebyshev coefficients,
the covariance matrix, fit diagnostics, and monomial coefficients normalized
to the SM point.

To print the old-style coefficient blocks directly:

```bash
multihiggs fit configs/gg_hhh_c3d4.toml --print-polynomial
```

That prints absolute Chebyshev coefficients, Chebyshev coefficients normalized
to the constant term, monomial coefficients in fitted variables such as
`k3,k4`, and the equivalent polynomial in scan variables such as `c3,d4`.

## Adapting To A New Process Or Model

Copy one of the config files and edit:

- `[process]`: `name`, `mg5_path`, `model`, `generate`, and `output`
- `[[couplings]]`: one entry per MadGraph parameter to scan
- `[scan]`: run number, collider energy, events per point, and integration options
- `[fit]`: Chebyshev terms to fit

By default `[scan]` has `no_cuts = true`, so the launch card clears common
MadGraph run-card cuts. Keep this for inclusive cross-section fits.

The event settings in `[scan]` control both generation and completed-run
detection:

```toml
[scan]
nevents = 10000
# Optional. If omitted, completed-run detection uses nevents.
# Use 0 to accept any existing LHE file regardless of event count.
min_events = 10000
```

For a fast cross-section-only integration scan where existing one-event samples
should count as complete, set:

```toml
[scan]
nevents = 1
min_events = 1
```

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

`hist` also uses the configured event-count minimum. With the starter configs,
that means only 10k-event runs are histogrammed. You can override the threshold
from the command line:

```bash
multihiggs hist configs/gg_hhhh_c3d4.toml --min-events 10000
```

The same explicit filter is available for `fit` when the input CSV contains an
`event_count` column:

```bash
multihiggs fit configs/gg_hhhh_c3d4.toml \
  --input outputs/gg_4h/histograms.csv \
  --min-events 10000
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

## References

- Original multi-Higgs loop-model repository:
  `https://gitlab.com/apapaefs/multihiggs_loop_sm`
- Andreas Papaefstathiou and Gilberto Tetlalmatzi-Xolocotzi,
  "Multi-Higgs boson production with anomalous interactions at current and
  future proton colliders", JHEP 06 (2024) 124,
  arXiv:2312.13562, doi:10.1007/JHEP06(2024)124.
