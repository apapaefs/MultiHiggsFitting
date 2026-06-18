# MultiHiggsFitting

Reusable MadGraph tooling for scanning multi-Higgs production over anomalous
couplings, collecting cross sections or LHE histograms, and fitting compact
polynomial bases.

The main use case is loop-induced multi-Higgs production with either:

- the original `loop_sm_c3d4` model, where MadGraph parameters are `c3,d4`;
- the `heft_loop_sm_restricted5` model, where the MHEFT deviations include
  `CT1, CT2, CT3, D3, D4`.

The shipped `c3,d4` configs scan the MadGraph parameters but fit the physical
kappa variables

```text
K3 = 1 + c3
K4 = 1 + d4
```

The restricted5 configs can use the same physical-basis machinery with

```text
KT = 1 + CT1
K3 = 1 + D3
K4 = 1 + D4
```

Non-SM contact variables such as `CT2` and `CT3` remain explicit fit
variables.

## Install

From this repository:

```bash
python3 -m pip install -e .
```

For plotting commands:

```bash
python3 -m pip install -e '.[plot]'
```

For ad hoc use without installing, run commands through the module:

```bash
python3 -m multihiggs grid configs/gg_hhh_c3d4.toml
```

The package requires Python 3.10 or newer plus the dependencies in
`pyproject.toml`.

## MadGraph Setup

Install the basic build tools used by MG5/aMC and generated matrix elements:

```bash
sudo apt-get update
sudo apt-get install -y python3 gfortran g++ make wget tar gzip
```

Unpack MG5/aMC at the repository root, or edit `mg5_path` in the config you
are using:

```bash
tar -xf LTS_MG5aMC_v3.5.15.tar
```

The example configs assume:

```toml
mg5_path = "MG5_aMC_v3_5_15"
```

Install the UFO models into the MG5 model directory. For a repo-local MG5
checkout:

```bash
ln -s ../../models/loop_sm_c3d4 MG5_aMC_v3_5_15/models/loop_sm_c3d4
ln -s ../../models/heft_loop_sm_restricted5 MG5_aMC_v3_5_15/models/heft_loop_sm_restricted5
```

The restricted5 model is vendored from
`https://gitlab.com/apapaefs/multihiggs_loop_sm` at commit `99ba5ee90669`.
If you are reproducing the old 4H area and prefer the archived model, unpack
the old tarball instead:

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

Loop-induced runs need MG5 loop libraries at runtime. The pipeline adds common
MG5 `HEPTools` and `COLLIER` library paths under `mg5_path` to
`LD_LIBRARY_PATH` when it launches MG5 or madevent.

## Configs

Primary `loop_sm_c3d4` configs:

- [configs/gg_hhh_c3d4.toml](configs/gg_hhh_c3d4.toml): `gg -> h h h`
- [configs/gg_hhhh_c3d4.toml](configs/gg_hhhh_c3d4.toml): `gg -> h h h h`

Light validation configs:

- [configs/gg_hh_c3_validation.toml](configs/gg_hh_c3_validation.toml):
  `gg -> h h [noborn=QCD]`
- [configs/gg_tthh_c3_validation.toml](configs/gg_tthh_c3_validation.toml):
  `gg -> t t~ h h`
- [configs/gg_tthhh_c3d4_validation.toml](configs/gg_tthhh_c3d4_validation.toml):
  `gg -> t t~ h h h`
- [configs/gg_tthh_restricted5_ct1_ct2_d3_validation.toml](configs/gg_tthh_restricted5_ct1_ct2_d3_validation.toml):
  restricted5 `gg -> t t~ h h` with `CT1, CT2, D3`
- [configs/gg_tthhh_restricted5_ct_ct2_c3_validation.toml](configs/gg_tthhh_restricted5_ct_ct2_c3_validation.toml):
  restricted5 `gg -> t t~ h h h` with `CT1, CT2, D3`
- [configs/gg_hhh_restricted5_ct_c3_d4_validation.toml](configs/gg_hhh_restricted5_ct_c3_d4_validation.toml):
  restricted5 `gg -> h h h [noborn=QCD]` with `CT1, D3, D4`
- [configs/gg_hhh_restricted5_ct_ct2_ct3_c3_d4_validation.toml](configs/gg_hhh_restricted5_ct_ct2_ct3_c3_d4_validation.toml):
  restricted5 `gg -> h h h [noborn=QCD]` with `CT1, CT2, CT3, D3, D4`
- [configs/gg_hhh_restricted5_ct1_ct2_ct3_d3_d4_validation.toml](configs/gg_hhh_restricted5_ct1_ct2_ct3_d3_d4_validation.toml):
  same process with the explicit uppercase parameter names in the scan labels
- [configs/gg_hhhh_restricted5_c3_d4_validation.toml](configs/gg_hhhh_restricted5_c3_d4_validation.toml):
  restricted5 `gg -> h h h h [noborn=QCD]` with `D3, D4`
- [configs/gg_tthhh_restricted5_ct_c3_d4_validation.toml](configs/gg_tthhh_restricted5_ct_c3_d4_validation.toml):
  restricted5 `gg -> t t~ h h h` with `CT1, D3, D4`

## Quick Workflow

This is the concise end-to-end path for a run-number campaign. The
`--mheft-basis sm-like` step asks the generated matrix code for the compact
physical kappa basis before fitting.

```bash
CONFIG=configs/gg_hhh_c3d4.toml
RUN=2

multihiggs grid "$CONFIG" --run-number "$RUN"
multihiggs generate-process "$CONFIG" --run
multihiggs infer-terms "$CONFIG" --mheft-basis sm-like
multihiggs scan "$CONFIG" --run-number "$RUN" --run
multihiggs collect "$CONFIG" --run-number "$RUN"
multihiggs fit "$CONFIG" --print-polynomial
```

What each command does:

- `grid`: writes `outputs/<process>/scan_points.csv`.
- `generate-process`: writes or runs the MG5 process card.
- `infer-terms`: reads the generated HELAS/MadLoop code and updates
  `[fit].terms`.
- `scan`: writes or runs the madevent launch card.
- `collect`: writes the collected cross sections to `xsecs.csv`.
- `fit`: writes `fit.json` and, with `--print-polynomial`, prints a readable
  polynomial report.

Use `--no-update-config` with `infer-terms` when you only want to inspect the
inferred basis:

```bash
multihiggs infer-terms "$CONFIG" --mheft-basis sm-like --no-update-config
```

## Scans And Run Selection

The scan grid is controlled by `[[couplings]]` blocks. A coupling can scan one
variable while reporting/fitting a shifted one:

```toml
[[couplings]]
name = "c3"
parameter = "c3"
fit_name = "k3"
range = [0.0, 2.0]
points = 7
fit_offset = 1.0
sm_value = 0.0
```

Here the grid is built in `k3 = 1+c3`, then converted back to the MadGraph
parameter `c3`. If `fit_offset = 0.0`, the grid is built directly in the scan
parameter.

With `sort = "sm_first"`, scan points are ordered by closeness to the SM point.
For a wide `d4` range this means many `d4 = 0` points can appear first even
though the full grid includes nonzero `d4` points.

Completed runs are skipped when
`Events/<run_name>/unweighted_events.lhe.gz` exists and has at least the
configured event minimum. By default this minimum is `[scan].nevents`. Override
it for quick or legacy samples:

```bash
multihiggs collect "$CONFIG" --run-number "$RUN" --min-events 1
```

Useful scan options:

```bash
multihiggs scan "$CONFIG" --max-runs 3
multihiggs scan "$CONFIG" --force
multihiggs scan "$CONFIG" --run-number 3 --run
```

Generated madevent cards disable common run-card cuts by default:

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

If MadEvent reports the known zero-amplitude multichannel failure, the pipeline
warns, skips that point, and continues with the remaining selected points:

```text
Problem in the multi-channeling. All amp2 are zero but not the total matrix-element
```

The command exits nonzero after continuing so the incomplete scan is not
mistaken for a fully successful run.

## Physical Bases And Term Inference

The inferred fit basis comes from the generated matrix-element code. The most
useful mode is:

```bash
multihiggs infer-terms "$CONFIG" --mheft-basis sm-like
```

`--mheft-basis sm-like` is available for both `infer-terms` and `fit`. It is
equivalent to:

```text
--term-map mheft_kappa --physical-basis --amplitude-basis sm_like
```

It does not change the scan grid or the MadGraph parameter values. It changes
the fit basis to the mapped physical variables:

```text
KT = 1 + CT1
K3 = 1 + D3  or  K3 = 1 + c3
K4 = 1 + D4  or  K4 = 1 + d4
```

With `infer-terms`, this writes:

```toml
[fit]
basis = "physical_monomial"
term_map = "mheft_kappa"
```

unless `--no-update-config` is used. With `fit`, it infers and uses the
physical basis for that fit only, leaving the input TOML unchanged.

The compact SM-like amplitude structures are:

- HH/ttHH: `KT^2`, `KT*K3`
- HHH: `KT^3`, `KT^2*K3`, `KT*K3^2`, `KT*K4`
- HHHH: `KT^4`, `KT^3*K3`, `KT^2*K3^2`, `KT*K3^3`,
  `KT^2*K4`, `KT*K3*K4`

Missing SM-like kappas are treated as fixed factors. For example, in the
`loop_sm_c3d4` model there is no scanned `KT`, so the HHH basis is projected
onto the available `K3,K4` variables. Contact terms such as `CT2` and `CT3`
are preserved explicitly.

To print a mapped polynomial without changing the fitted basis, use the lower
level map option:

```bash
multihiggs infer-terms "$CONFIG" --no-update-config --term-map mheft_kappa
```

Add `--expand-term-map` to also see the fully expanded support in the mapped
variables. Expanded output can contain more monomials than independent fitted
coefficients.

For restricted5 processes, generated MG5 process cards and inferred
cross-section terms are restricted by the maximum squared MHEFT order. If the
process string already contains a constraint such as `MHEFT^2<=6`, that value
is used. Otherwise the default cap is twice the number of final-state Higgs
bosons in the primary process: `MHEFT^2<=4` for `hh`, `MHEFT^2<=6` for `hhh`,
`MHEFT^2<=8` for `hhhh`, and so on. For loop-induced restricted5 processes
with `[noborn=QCD]`, the generated card normalizes this to:

```text
[noborn=QCD MHEFT] MHEFT^2<=N
```

Term inference is a generated-code cross-check, not a proof. Cancellations or
model-specific details can still remove or add practical fit requirements, so
the command prints warnings to check the inferred terms before production use.

## Fit Output

For physical-monomial configs such as `gg_hhh_c3d4`, `fit.json` stores fitted
coefficients in the mapped variables, for example `K3,K4`, and also includes
the equivalent polynomial in the original scan variables such as `c3,d4`.

Print a readable report with:

```bash
multihiggs fit "$CONFIG" --print-polynomial
```

For a Chebyshev config, the report also includes the absolute and normalized
Chebyshev coefficients plus transformed monomial coefficients. For a
physical-monomial config, the report starts directly from the fitted physical
coefficients.

## Histograms And Distributions

Configs can define `[[observables]]` for simple LHE histograms:

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

Fit every histogram bin with the same configured polynomial basis:

```bash
multihiggs fit configs/gg_hhh_c3d4.toml \
  --input outputs/gg_hhh/histograms.csv \
  --output outputs/gg_hhh/hist_fit.json
```

Plot fitted distributions:

```bash
multihiggs distribution configs/gg_hhh_c3d4.toml \
  --input outputs/gg_hhh/hist_fit.json \
  --observable h_pt \
  --point c3=5,d4=100 \
  --point c3=-5,d4=-100 \
  --output outputs/gg_hhh/h_pt_distribution.png \
  --pdf-output outputs/gg_hhh/h_pt_distribution.pdf
```

The SM distribution is included by default. Use `--no-sm` to omit it. Repeat
`--point` to overlay several coupling points; unspecified couplings are fixed
to their configured `sm_value`. Use `--density` to plot bin contents divided by
bin width and `--log-y` for a logarithmic y axis.

For per-object observables such as `h_pt` with `which = "all"`, the histogram
integral is multiplicity times the event cross section.

## Contours And Variations

Plot a normalized two-variable contour:

```bash
multihiggs contour configs/gg_hhhh_c3d4.toml --x c3 --y d4
```

By default this reads `outputs/<process>/fit.json`, selects the `xsec` fit if
present, and scans each axis over the configured fit range converted back to
the scan variable. The axis names can be the coupling `name`, MadGraph
`parameter`, or `fit_name`.

Useful contour options:

```bash
multihiggs contour configs/gg_hhhh_c3d4.toml \
  --x c3 --y d4 \
  --x-range -30 30 --y-range -700 700 \
  --output outputs/gg_4h/hhhh_c3_d4_contour.png \
  --pdf-output outputs/gg_4h/hhhh_c3_d4_contour.pdf
```

Plot a one-dimensional ratio:

```bash
multihiggs variation configs/gg_hhhh_c3d4.toml --x c3
```

Fix non-axis couplings with repeatable `--fix name=value` arguments:

```bash
multihiggs variation configs/gg_hhhh_c3d4.toml \
  --x c3 \
  --fix d4=100 \
  --x-range -30 30 \
  --points 301 \
  --output outputs/gg_4h/hhhh_c3_variation_d4_100.png \
  --pdf-output outputs/gg_4h/hhhh_c3_variation_d4_100.pdf
```

For histogram fits, pass the histogram fit JSON and select the bin label:

```bash
multihiggs contour configs/gg_hhh_c3d4.toml \
  --input outputs/gg_hhh/hist_fit.json \
  --label h_pt:bin0 \
  --x c3 --y d4
```

## Adapting A Config

Copy a nearby config and edit:

- `[process]`: `name`, `mg5_path`, `model`, `generate`, and `output`
- `[[couplings]]`: one entry per MadGraph parameter to scan
- `[scan]`: run number, collider energy, events per point, sort mode, and
  integration options
- `[fit]`: `basis`, `term_map` when needed, and `terms`
- `[[observables]]`: optional LHE histogram definitions

Custom term maps can be defined in the TOML:

```toml
[term_maps.custom_top]
description = "Custom top modifier"

[[term_maps.custom_top.variables]]
source = "CT1"
name = "YT"
offset = 1.0
```

The `source` value is matched against the MadGraph/UFO parameter name first,
with config names and fit names also accepted for fitted polynomial output.

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
