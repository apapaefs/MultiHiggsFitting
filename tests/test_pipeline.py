from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multihiggs.config import load_config
from multihiggs.fit import fit_values
from multihiggs.grid import generate_scan_points
from multihiggs.histograms import histogram_lhe
from multihiggs.results import read_xsec


ROOT = Path(__file__).resolve().parents[1]


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

    def test_fit_recovers_synthetic_polynomial(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray([2.0 + 0.5 * c3 - 0.25 * d4 + 0.1 * c3 * c3 for c3, d4 in values])
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")
        self.assertEqual(result.rank, len(config.fit.terms))
        self.assertAlmostEqual(result.sigma_sm, 2.0, places=8)

    def test_read_xsec_from_lhe_gzip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.lhe.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("<LesHouchesEvents>\n")
                stream.write("#  Integrated weight (pb)  :  1.234e-05\n")
                stream.write("</LesHouchesEvents>\n")
            self.assertAlmostEqual(read_xsec(path), 1.234e-05)

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
            self.assertGreater(err.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
