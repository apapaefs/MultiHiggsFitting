from __future__ import annotations

import contextlib
import gzip
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from multihiggs.config import ScanConfig, load_config
from multihiggs.fit import fit_values, format_polynomial_report
from multihiggs.grid import generate_scan_points
from multihiggs.histograms import histogram_lhe
from multihiggs.madgraph import build_launch_block, write_madevent_card
from multihiggs.results import event_count, read_xsec


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
