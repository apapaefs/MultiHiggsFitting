from __future__ import annotations

import contextlib
import gzip
import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multihiggs.cli import main
from multihiggs.config import ScanConfig, load_config
from multihiggs.fit import fit_values, format_polynomial_report
from multihiggs.grid import generate_scan_points
from multihiggs.histograms import histogram_lhe
from multihiggs.madgraph import (
    build_launch_block,
    mg_runtime_env,
    patch_macos_madloop_rpaths,
    run_madevent,
    run_madevent_points,
    run_mg5,
    write_madevent_card,
)
from multihiggs.results import event_count, read_xsec
from multihiggs.term_inference import format_inferred_terms, infer_terms_from_process_dir


ROOT = Path(__file__).resolve().parents[1]


def write_test_lhe(path: Path, n_events: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        stream.write("<LesHouchesEvents>\n")
        for _ in range(n_events):
            stream.write("<event>\n")
            stream.write("0 1 1.0 1.0 1.0 1.0\n")
            stream.write("</event>\n")
        stream.write("#  Integrated weight (pb)  :  1.0\n")
        stream.write("</LesHouchesEvents>\n")


def write_generated_process_fixture(process_dir: Path) -> None:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    matrix_dir = process_dir / "SubProcesses" / "P1_fixture"
    model_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    (model_dir / "couplings.py").write_text(
        """
GC_SM = Coupling(name = 'GC_SM',
                 value = 'yt',
                 order = {'QED':1})
GC_C3 = Coupling(name = 'GC_C3',
                 value = '-6*complex(0,1)*lam*v*(1 + c3)',
                 order = {'QED':1})
GC_D4 = Coupling(name = 'GC_D4',
                 value = '-6*complex(0,1)*lam*(1 + d4)',
                 order = {'QED':2})
""",
        encoding="utf-8",
    )
    (matrix_dir / "matrix1_orig.f").write_text(
        """
      SUBROUTINE MATRIX1()
      CALL SXXXXX(P(0,1),+1*IC(1),W(1,1))
      CALL SXXXXX(P(0,2),+1*IC(2),W(1,2))
      CALL SSS1_1(W(1,1),W(1,2),GC_C3,MDL_MH,FK_MDL_WH,W(1,3))
      CALL SSS1_1(W(1,1),W(1,2),GC_D4,MDL_MH,FK_MDL_WH,W(1,4))
      CALL SSS1_0(W(1,1),W(1,2),GC_SM,AMP(1))
      CALL SSS1_0(W(1,3),W(1,2),GC_SM,AMP(2))
      CALL SSS1_0(W(1,4),W(1,2),GC_SM,AMP(3))
      END
""",
        encoding="utf-8",
    )


class PipelineTests(unittest.TestCase):
    def test_hhh_grid_matches_expected_size_and_sm_first(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        self.assertEqual(len(points), 41)
        self.assertEqual(points[0].values, (0.0, 0.0))
        self.assertEqual(points[0].texts, ("0.0", "0.0"))

    def test_hhhh_grid_matches_expected_size_and_terms(self):
        config = load_config(ROOT / "configs" / "gg_hhhh_c3d4.toml")
        points = generate_scan_points(config)
        self.assertEqual(len(points), 41)
        self.assertEqual(len(config.fit.terms), 15)

    def test_hh_validation_grid_scans_c3_directly(self):
        config = load_config(ROOT / "configs" / "gg_hh_c3_validation.toml")
        points = generate_scan_points(config)
        values = [point.values[0] for point in points]
        self.assertEqual(config.generate, "g g > h h [noborn=QCD]")
        self.assertEqual(len(points), 5)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(min(values), -20.0)
        self.assertEqual(max(values), 20.0)
        self.assertEqual(config.couplings[0].fit_name, "c3")
        self.assertEqual(config.couplings[0].fit_offset, 0.0)
        self.assertEqual(config.fit.terms, ((0,), (1,), (2,)))

    def test_tthh_validation_grid_scans_c3_directly(self):
        config = load_config(ROOT / "configs" / "gg_tthh_c3_validation.toml")
        points = generate_scan_points(config)
        values = [point.values[0] for point in points]
        self.assertEqual(len(points), 5)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(min(values), -20.0)
        self.assertEqual(max(values), 20.0)
        self.assertEqual(config.couplings[0].fit_name, "c3")
        self.assertEqual(config.couplings[0].fit_offset, 0.0)
        self.assertEqual(config.fit.terms, ((0,), (1,), (2,)))

    def test_tthhh_validation_grid_scans_c3_d4_directly(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_c3d4_validation.toml")
        points = generate_scan_points(config)
        c3_values = [point.values[0] for point in points]
        d4_values = [point.values[1] for point in points]
        self.assertEqual(config.generate, "g g > t t~ h h h")
        self.assertEqual(len(points), 15)
        self.assertEqual(points[0].values, (0.0, 0.0))
        self.assertEqual(min(c3_values), -20.0)
        self.assertEqual(max(c3_values), 20.0)
        self.assertEqual(min(d4_values), -100.0)
        self.assertEqual(max(d4_values), 100.0)
        self.assertEqual(config.couplings[0].fit_offset, 0.0)
        self.assertEqual(config.couplings[1].fit_offset, 0.0)
        self.assertEqual(
            config.fit.terms,
            (
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
            ),
        )

    def test_restricted5_tthhh_validation_maps_readable_modifier_names(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")
        points = generate_scan_points(config)
        self.assertEqual(config.model, "heft_loop_sm_restricted5")
        self.assertEqual(config.generate, "g g > t t~ h h h")
        self.assertEqual(len(points), 27)
        self.assertEqual(points[0].values, (0.0, 0.0, 0.0))
        self.assertEqual([coupling.name for coupling in config.couplings], ["ct", "ct2", "c3"])
        self.assertEqual([coupling.parameter for coupling in config.couplings], ["CT1", "CT2", "D3"])
        self.assertEqual(config.couplings[0].fit_range, (-0.5, 0.5))
        self.assertEqual(config.scan.extra_set_commands, ["set CT3 0.0", "set D4 0.0"])
        self.assertEqual(len(config.fit.terms), 27)

    def test_scan_defaults_to_ten_thousand_events(self):
        scan = ScanConfig.from_dict({})
        self.assertEqual(scan.nevents, 10000)
        self.assertEqual(scan.event_minimum, 10000)
        scan = ScanConfig.from_dict({"nevents": 5000})
        self.assertEqual(scan.event_minimum, 5000)
        scan = ScanConfig.from_dict({"nevents": 5000, "min_events": 1000})
        self.assertEqual(scan.event_minimum, 1000)

    def test_fit_recovers_synthetic_polynomial(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray([2.0 + 0.5 * c3 - 0.25 * d4 + 0.1 * c3 * c3 for c3, d4 in values])
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")
        self.assertEqual(result.rank, len(config.fit.terms))
        self.assertAlmostEqual(result.sigma_sm, 2.0, places=8)

    def test_polynomial_report_includes_original_blocks(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray([2.0 + 0.5 * c3 - 0.25 * d4 + 0.1 * c3 * c3 for c3, d4 in values])
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")
        report = format_polynomial_report(config, result)
        self.assertIn("Absolute Chebyshev coefficients:", report)
        self.assertIn("Physical monomial coefficients in k3,k4:", report)
        self.assertIn("Physical polynomial coefficients in c3,d4:", report)

    def test_madevent_launch_block_disables_common_cuts(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        point = generate_scan_points(config)[0]
        block = "\n".join(build_launch_block(config, point))
        self.assertIn("set dsqrt_shat 0.0", block)
        self.assertIn("set ptheavy 0.0", block)
        self.assertIn("set pt_min_pdg {}", block)
        self.assertIn("set eta_max_pdg {}", block)
        self.assertIn("set mxx_min_pdg {}", block)
        self.assertIn("set nevents 10000", block)

    def test_run_commands_accept_relative_mg5_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5_aMC_v3_5_15"
            process_dir = mg5_path / "proc"
            (mg5_path / "bin").mkdir(parents=True)
            (process_dir / "bin").mkdir(parents=True)

            mg5_bin = mg5_path / "bin" / "mg5_aMC"
            mg5_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            mg5_bin.chmod(0o755)

            madevent_bin = process_dir / "bin" / "madevent"
            madevent_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            madevent_bin.chmod(0o755)

            process_card = tmp_root / "process.mg5"
            process_card.write_text("quit\n", encoding="utf-8")
            command_file = tmp_root / "scan.dcmd"
            command_file.write_text("", encoding="utf-8")

            config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
            config = replace(
                config,
                mg5_path=mg5_path.relative_to(ROOT),
                output="proc",
            )

            self.assertEqual(run_mg5(config, process_card), 0)
            self.assertEqual(run_madevent(config, command_file), 0)

    def test_mg_runtime_env_sets_macos_dyld_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mg5_path = Path(tmpdir) / "MG5"
            heptools_lib = mg5_path / "HEPTools" / "lib"
            heptools_lib.mkdir(parents=True)

            with patch("multihiggs.madgraph.sys.platform", "darwin"), patch.dict(os.environ, {}, clear=True):
                env = mg_runtime_env(mg5_path)

            self.assertIn(str(heptools_lib), env["LD_LIBRARY_PATH"].split(":"))
            self.assertIn(str(heptools_lib), env["DYLD_LIBRARY_PATH"].split(":"))

    def test_mg_runtime_env_keeps_linux_environment_linux_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mg5_path = Path(tmpdir) / "MG5"
            heptools_lib = mg5_path / "HEPTools" / "lib"
            heptools_lib.mkdir(parents=True)

            with patch("multihiggs.madgraph.sys.platform", "linux"), patch.dict(os.environ, {}, clear=True):
                env = mg_runtime_env(mg5_path)

            self.assertIn(str(heptools_lib), env["LD_LIBRARY_PATH"].split(":"))
            self.assertNotIn("DYLD_LIBRARY_PATH", env)

    def test_macos_madloop_makefile_links_with_rpath_only_on_darwin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            process_dir = Path(tmpdir)
            subproc_dir = process_dir / "SubProcesses"
            subproc_dir.mkdir()
            makefiles = [subproc_dir / "makefile", subproc_dir / "makefile_MadLoop"]
            for makefile in makefiles:
                makefile.write_text(
                    "LINKLIBS = -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS) $(LDFLAGS)\n",
                    encoding="utf-8",
                )

            with patch("multihiggs.madgraph.sys.platform", "linux"):
                patch_macos_madloop_rpaths(process_dir)
            for makefile in makefiles:
                self.assertNotIn("$(RPATH_LIBS)", makefile.read_text(encoding="utf-8"))

            with patch("multihiggs.madgraph.sys.platform", "darwin"):
                patch_macos_madloop_rpaths(process_dir)
                patch_macos_madloop_rpaths(process_dir)

            for makefile in makefiles:
                text = makefile.read_text(encoding="utf-8")
                self.assertEqual(text.count("$(RPATH_LIBS)"), 1)
                self.assertIn("$(LDFLAGS) $(RPATH_LIBS)", text)

    def test_madevent_points_warns_and_continues_after_zero_amplitude_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5_aMC_v3_5_15"
            process_dir = mg5_path / "proc"
            (process_dir / "bin").mkdir(parents=True)

            config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")
            config = replace(config, mg5_path=mg5_path.relative_to(ROOT), output="proc")
            bad_point = replace(generate_scan_points(config)[0], values=(-1.0, 0.0, 0.0), texts=("-1.0", "0.0", "0.0"))
            good_point = replace(generate_scan_points(config)[0], values=(0.5, 0.0, 0.0), texts=("0.5", "0.0", "0.0"))

            madevent_bin = process_dir / "bin" / "madevent"
            madevent_bin.write_text(
                f"""#!/bin/sh
grep '^launch ' "$1" >> attempts.log
if grep -q 'set CT1 -1.0' "$1"; then
  cat > {bad_point.run_name(config)}_tag_1_debug.log <<'EOF'
Problem in the multi-channeling. All amp2 are zero but not the total matrix-element
EOF
  exit 1
fi
exit 0
""",
                encoding="utf-8",
            )
            madevent_bin.chmod(0o755)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = run_madevent_points(config, [bad_point, good_point], tmp_root / "scan.dcmd")

            self.assertEqual(status, 2)
            output = stdout.getvalue()
            self.assertIn("zero-amplitude multichannel issue", output)
            self.assertIn("CT1 = -1 cancels the SM ttH Yukawa", output)
            attempts = (process_dir / "attempts.log").read_text(encoding="utf-8")
            self.assertIn(bad_point.run_name(config), attempts)
            self.assertIn(good_point.run_name(config), attempts)

    def test_infers_terms_from_generated_matrix_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            process_dir = Path(tmpdir) / "proc"
            write_generated_process_fixture(process_dir)

            result = infer_terms_from_process_dir(process_dir, ("c3", "d4"))

            self.assertEqual(result.amplitude_terms, ((0, 0), (1, 0), (0, 1)))
            self.assertEqual(
                result.cross_section_terms,
                ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)),
            )
            self.assertEqual(result.amplitude_support_counts[((0, 0), (1, 0))], 1)
            self.assertEqual(result.amplitude_support_counts[((0, 0), (0, 1))], 1)

            report = format_inferred_terms(result)
            self.assertIn("Inferred cross-section [fit].terms:", report)
            self.assertIn("  [1, 1],", report)
            self.assertTrue(report.rstrip().endswith("WARNING: CHECK inferred fit terms before using them."))

    def test_infer_terms_command_prints_check_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_generated_process_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "loop_sm_c3d4"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "c3"
parameter = "c3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 3

[[couplings]]
name = "d4"
parameter = "d4"
fit_name = "d4"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [[0, 0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(["infer-terms", str(config_path)])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("terms = [", output)
            self.assertIn("  [2, 0],", output)
            self.assertTrue(output.rstrip().endswith("WARNING: CHECK inferred fit terms before using them."))

    def test_scan_reschedules_existing_runs_below_configured_event_minimum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
            config = replace(
                config,
                name="test",
                mg5_path=Path(tmpdir),
                output="proc",
                scan=replace(config.scan, nevents=2, min_events=None),
            )
            point = generate_scan_points(config)[0]
            lhe_file = config.process_dir / "Events" / point.run_name(config) / "unweighted_events.lhe.gz"
            card = Path(tmpdir) / "scan.dcmd"

            write_test_lhe(lhe_file, 1)
            with contextlib.redirect_stdout(io.StringIO()):
                _, selected = write_madevent_card(config, [point], card)
            self.assertEqual(selected, [point])

            write_test_lhe(lhe_file, 2)
            with contextlib.redirect_stdout(io.StringIO()):
                _, selected = write_madevent_card(config, [point], card)
            self.assertEqual(selected, [])

    def test_read_xsec_from_lhe_gzip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.lhe.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("<LesHouchesEvents>\n")
                stream.write("<event>\n")
                stream.write("0 1 1.0 1.0 1.0 1.0\n")
                stream.write("</event>\n")
                stream.write("<event>\n")
                stream.write("0 1 1.0 1.0 1.0 1.0\n")
                stream.write("</event>\n")
                stream.write("#  Integrated weight (pb)  :  1.234e-05\n")
                stream.write("</LesHouchesEvents>\n")
            self.assertAlmostEqual(read_xsec(path), 1.234e-05)
            self.assertEqual(event_count(path), 2)

    def test_histogram_lhe_higgs_pt(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        observable = config.observables[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.lhe.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("<LesHouchesEvents>\n")
                stream.write("#  Integrated weight (pb)  :  3.0\n")
                stream.write("<event>\n")
                stream.write("3 1 1.0 1.0 1.0 1.0\n")
                stream.write("25 1 0 0 0 0 30 40 0 125 125 0 0\n")
                stream.write("25 1 0 0 0 0 60 80 0 125 125 0 0\n")
                stream.write("25 1 0 0 0 0 0 0 0 125 125 0 0\n")
                stream.write("</event>\n")
                stream.write("</LesHouchesEvents>\n")
            hist, err = histogram_lhe(path, observable)
            self.assertAlmostEqual(hist.sum(), 9.0)
            self.assertTrue(np.all(err > 0.0))


if __name__ == "__main__":
    unittest.main()
