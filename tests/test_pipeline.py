from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multihiggs.cli import main
from multihiggs.config import ScanConfig, load_config
from multihiggs.contours import (
    PLOT_TITLE_FONTSIZE,
    build_contour_data,
    build_variation_data,
    parse_fixed_values,
)
from multihiggs.distributions import (
    DistributionData,
    DistributionSeries,
    build_distribution_data,
    draw_distribution_series,
    distribution_series_style,
    histogram_step_xy,
    parse_parameter_points,
)
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
        self.assertEqual(len(points), 105)
        self.assertEqual(points[0].values, (0.0, 0.0, 0.0))
        self.assertEqual([coupling.name for coupling in config.couplings], ["ct", "ct2", "c3"])
        self.assertEqual([coupling.parameter for coupling in config.couplings], ["CT1", "CT2", "D3"])
        self.assertEqual([coupling.points for coupling in config.couplings], [7, 3, 5])
        self.assertEqual(config.couplings[0].fit_range, (-0.5, 0.5))
        self.assertEqual(config.scan.madgraph.accuracy, 0.01)
        self.assertEqual(config.scan.madgraph.points, 1000)
        self.assertEqual(config.scan.madgraph.iterations, 5)
        self.assertEqual(config.scan.extra_set_commands, ["set CT3 0.0", "set D4 0.0"])
        self.assertEqual(len(config.fit.terms), 45)
        self.assertEqual(max(term[0] for term in config.fit.terms), 6)
        self.assertEqual(max(term[1] for term in config.fit.terms), 2)
        self.assertEqual(max(term[2] for term in config.fit.terms), 4)

    def test_restricted5_hhh_ct_c3_d4_validation_config(self):
        config = load_config(ROOT / "configs" / "gg_hhh_restricted5_ct_c3_d4_validation.toml")
        points = generate_scan_points(config)
        self.assertEqual(config.model, "heft_loop_sm_restricted5")
        self.assertEqual(config.generate, "g g > h h h [noborn=QCD]")
        self.assertEqual(len(points), 45)
        self.assertEqual(points[0].values, (0.0, 0.0, 0.0))
        self.assertEqual([coupling.name for coupling in config.couplings], ["ct", "c3", "d4"])
        self.assertEqual([coupling.parameter for coupling in config.couplings], ["CT1", "D3", "D4"])
        self.assertEqual([coupling.points for coupling in config.couplings], [5, 3, 3])
        self.assertEqual(config.scan.extra_set_commands, ["set CT2 0.0", "set CT3 0.0"])
        self.assertGreater(len(config.fit.terms), 0)
        self.assertLessEqual(len(config.fit.terms), len(points))
        self.assertIn((0, 0, 0), config.fit.terms)
        self.assertTrue(all(len(term) == 3 for term in config.fit.terms))

    def test_restricted5_hhh_five_modifier_validation_config(self):
        config = load_config(ROOT / "configs" / "gg_hhh_restricted5_ct_ct2_ct3_c3_d4_validation.toml")
        points = generate_scan_points(config)
        self.assertEqual(config.model, "heft_loop_sm_restricted5")
        self.assertEqual(config.generate, "g g > h h h [noborn=QCD]")
        self.assertEqual(len(points), 243)
        self.assertEqual(points[0].values, (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual([coupling.name for coupling in config.couplings], ["ct", "ct2", "ct3", "c3", "d4"])
        self.assertEqual([coupling.parameter for coupling in config.couplings], ["CT1", "CT2", "CT3", "D3", "D4"])
        self.assertEqual([coupling.points for coupling in config.couplings], [3, 3, 3, 3, 3])
        self.assertEqual(config.scan.extra_set_commands, [])
        self.assertGreater(len(config.fit.terms), 0)
        self.assertLessEqual(len(config.fit.terms), len(points))
        self.assertIn((0, 0, 0, 0, 0), config.fit.terms)
        self.assertTrue(all(len(term) == 5 for term in config.fit.terms))

    def test_restricted5_hhhh_c3_d4_validation_config(self):
        config = load_config(ROOT / "configs" / "gg_hhhh_restricted5_c3_d4_validation.toml")
        points = generate_scan_points(config)
        self.assertEqual(config.model, "heft_loop_sm_restricted5")
        self.assertEqual(config.generate, "g g > h h h h [noborn=QCD]")
        self.assertEqual(len(points), 9)
        self.assertEqual(points[0].values, (0.0, 0.0))
        self.assertEqual([coupling.name for coupling in config.couplings], ["c3", "d4"])
        self.assertEqual([coupling.parameter for coupling in config.couplings], ["D3", "D4"])
        self.assertEqual(config.scan.extra_set_commands, ["set CT1 0.0", "set CT2 0.0", "set CT3 0.0"])
        self.assertGreater(len(config.fit.terms), 0)
        self.assertLessEqual(len(config.fit.terms), len(points))
        self.assertIn((0, 0), config.fit.terms)
        self.assertTrue(all(len(term) == 2 for term in config.fit.terms))

    def test_restricted5_tthhh_ct_c3_d4_validation_config(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_c3_d4_validation.toml")
        points = generate_scan_points(config)
        self.assertEqual(config.model, "heft_loop_sm_restricted5")
        self.assertEqual(config.generate, "g g > t t~ h h h")
        self.assertEqual(len(points), 45)
        self.assertEqual(points[0].values, (0.0, 0.0, 0.0))
        self.assertEqual([coupling.name for coupling in config.couplings], ["ct", "c3", "d4"])
        self.assertEqual([coupling.parameter for coupling in config.couplings], ["CT1", "D3", "D4"])
        self.assertEqual([coupling.points for coupling in config.couplings], [5, 3, 3])
        self.assertEqual(config.scan.extra_set_commands, ["set CT2 0.0", "set CT3 0.0"])
        self.assertGreater(len(config.fit.terms), 0)
        self.assertLessEqual(len(config.fit.terms), len(points))
        self.assertIn((0, 0, 0), config.fit.terms)
        self.assertTrue(all(len(term) == 3 for term in config.fit.terms))

    def test_scan_defaults_to_ten_thousand_events(self):
        scan = ScanConfig.from_dict({})
        self.assertEqual(scan.nevents, 10000)
        self.assertEqual(scan.event_minimum, 10000)
        scan = ScanConfig.from_dict({"nevents": 5000})
        self.assertEqual(scan.event_minimum, 5000)
        scan = ScanConfig.from_dict({"nevents": 5000, "min_events": 1000})
        self.assertEqual(scan.event_minimum, 1000)

    def test_grid_command_accepts_run_number_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "grid.csv"

            status = main([
                "grid",
                str(ROOT / "configs" / "gg_hh_c3_validation.toml"),
                "-o",
                str(output),
                "--run-number",
                "7",
            ])

            self.assertEqual(status, 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn("run_gg_hh_c3_validation_7_0.0", text)
            self.assertNotIn("run_gg_hh_c3_validation_1_0.0", text)

    def test_scan_command_accepts_run_number_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "scan.dcmd"

            status = main([
                "scan",
                str(ROOT / "configs" / "gg_hh_c3_validation.toml"),
                "-o",
                str(output),
                "--max-runs",
                "1",
                "--run-number",
                "7",
            ])

            self.assertEqual(status, 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn("launch run_gg_hh_c3_validation_7_0.0", text)
            self.assertNotIn("launch run_gg_hh_c3_validation_1_0.0", text)

    def test_fit_recovers_synthetic_polynomial(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray([2.0 + 0.5 * c3 - 0.25 * d4 + 0.1 * c3 * c3 for c3, d4 in values])
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")
        self.assertEqual(result.rank, len(config.fit.terms))
        self.assertAlmostEqual(result.sigma_sm, 2.0, places=8)

    def test_contour_data_evaluates_two_axes_with_fixed_third_variable(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray(
            [
                10.0 + 2.0 * ct + 3.0 * ct2 + 5.0 * c3 + 7.0 * ct * c3
                for ct, ct2, c3 in values
            ]
        )
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")

        contour = build_contour_data(
            config,
            result.to_dict(config),
            x_name="ct",
            y_name="c3",
            fixed_values={"ct2": 0.25},
            x_range=(-0.5, 0.5),
            y_range=(-1.0, 1.0),
            x_points=3,
            y_points=3,
        )

        self.assertEqual(contour.fixed_values, {"ct2": 0.25})
        for y_index, c3 in enumerate(contour.y_values):
            for x_index, ct in enumerate(contour.x_values):
                expected = (10.0 + 2.0 * ct + 3.0 * 0.25 + 5.0 * c3 + 7.0 * ct * c3) / 10.0
                self.assertAlmostEqual(contour.ratio[y_index, x_index], expected, places=8)

    def test_variation_data_scans_one_axis_with_fixed_values_and_sm_defaults(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray(
            [
                10.0 + 2.0 * ct + 3.0 * ct2 + 5.0 * c3 + 7.0 * ct * c3
                for ct, ct2, c3 in values
            ]
        )
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")

        variation = build_variation_data(
            config,
            result.to_dict(config),
            x_name="ct",
            fixed_values={"c3": 0.5},
            x_range=(-0.5, 0.5),
            points=3,
        )

        self.assertEqual(variation.x_name, "ct")
        self.assertEqual(variation.fixed_values, {"c3": 0.5})
        np.testing.assert_allclose(variation.x_values, [-0.5, 0.0, 0.5])
        for index, ct in enumerate(variation.x_values):
            expected = (10.0 + 2.0 * ct + 5.0 * 0.5 + 7.0 * ct * 0.5) / 10.0
            self.assertAlmostEqual(variation.ratio[index], expected, places=8)

    def test_fixed_values_reject_axis_overrides(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")

        with self.assertRaisesRegex(ValueError, "axis variable"):
            parse_fixed_values(config, ["ct=0.1"], axis_names={"ct", "c3"})

    def test_contour_plot_title_uses_smaller_font(self):
        self.assertEqual(PLOT_TITLE_FONTSIZE, 13)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_contour_command_writes_plot_with_fixed_override(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray(
            [
                10.0 + 2.0 * ct + 3.0 * ct2 + 5.0 * c3 + 7.0 * ct * c3
                for ct, ct2, c3 in values
            ]
        )
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fit_json = tmp_path / "fit.json"
            fit_json.write_text(
                json.dumps({"fits": [result.to_dict(config)]}),
                encoding="utf-8",
            )
            output = tmp_path / "ct_c3_contour.png"

            with patch.dict(os.environ, {"MPLCONFIGDIR": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path)}):
                status = main(
                    [
                        "contour",
                        str(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml"),
                        "--input",
                        str(fit_json),
                        "--x",
                        "ct",
                        "--y",
                        "c3",
                        "--fix",
                        "ct2=0.25",
                        "--output",
                        str(output),
                        "--points",
                        "5",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_variation_command_writes_plot_with_fixed_override(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        y = np.asarray(
            [
                10.0 + 2.0 * ct + 3.0 * ct2 + 5.0 * c3 + 7.0 * ct * c3
                for ct, ct2, c3 in values
            ]
        )
        yerr = np.ones(len(values)) * 0.01
        result = fit_values(config, values, y, yerr, "synthetic")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fit_json = tmp_path / "fit.json"
            fit_json.write_text(
                json.dumps({"fits": [result.to_dict(config)]}),
                encoding="utf-8",
            )
            output = tmp_path / "ct_variation.png"

            with patch.dict(os.environ, {"MPLCONFIGDIR": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path)}):
                status = main(
                    [
                        "variation",
                        str(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml"),
                        "--input",
                        str(fit_json),
                        "--x",
                        "ct",
                        "--fix",
                        "c3=0.5",
                        "--output",
                        str(output),
                        "--points",
                        "5",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_distribution_data_evaluates_histogram_bin_fits_at_points_and_sm(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        observable = config.observables[0]
        fit_records = []
        for bin_index in range(len(observable.bins) - 1):
            scale = float(bin_index + 1)
            y = np.asarray([scale * (10.0 + 2.0 * c3 - 0.1 * d4) for c3, d4 in values])
            yerr = np.ones(len(values)) * 0.01
            fit_records.append(fit_values(config, values, y, yerr, f"{observable.name}:bin{bin_index}").to_dict(config))

        distribution = build_distribution_data(
            config,
            {"fits": fit_records},
            observable_name=observable.name,
            parameter_points=[{"c3": 1.0, "d4": 2.0}, {"c3": -1.0}],
            include_sm=True,
        )

        self.assertEqual(distribution.observable.name, observable.name)
        self.assertEqual([series.label for series in distribution.series], ["c3=1, d4=2", "c3=-1", "SM"])
        for bin_index, value in enumerate(distribution.series[0].values):
            expected = float(bin_index + 1) * (10.0 + 2.0 * 1.0 - 0.1 * 2.0)
            self.assertAlmostEqual(value, expected, places=8)
        for bin_index, value in enumerate(distribution.series[1].values):
            expected = float(bin_index + 1) * (10.0 + 2.0 * -1.0)
            self.assertAlmostEqual(value, expected, places=8)
        for bin_index, value in enumerate(distribution.series[2].values):
            expected = float(bin_index + 1) * 10.0
            self.assertAlmostEqual(value, expected, places=8)
        self.assertTrue(np.all(distribution.series[0].errors >= 0.0))

    def test_histogram_step_xy_draws_continuous_outline_without_boxing_each_bin(self):
        x_values, y_values = histogram_step_xy(
            np.asarray([0.0, 10.0, 20.0]),
            np.asarray([10.0, 20.0, 30.0]),
            np.asarray([1.0, 2.0, 4.0]),
        )

        expected_x = [0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
        expected_y = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.0]
        np.testing.assert_allclose(
            x_values,
            expected_x,
        )
        np.testing.assert_allclose(
            y_values,
            expected_y,
        )

    def test_draw_distribution_series_uses_step_line_and_y_only_errorbars(self):
        class RecordingAxes:
            def __init__(self):
                self.plot_calls = []
                self.errorbar_calls = []

            def plot(self, *args, **kwargs):
                self.plot_calls.append((args, kwargs))

            def errorbar(self, *args, **kwargs):
                self.errorbar_calls.append((args, kwargs))

        observable = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml").observables[0]
        series = DistributionSeries(
            label="c3=1",
            values=np.asarray([1.0, 2.0, 3.0]),
            errors=np.asarray([0.1, 0.2, 0.3]),
            parameter_values={"c3": 1.0},
        )
        data = DistributionData(
            observable=observable,
            bin_lows=np.asarray([0.0, 10.0, 20.0]),
            bin_highs=np.asarray([10.0, 20.0, 30.0]),
            bin_centers=np.asarray([5.0, 15.0, 25.0]),
            bin_half_widths=np.asarray([5.0, 5.0, 5.0]),
            series=[series],
            density=False,
        )
        ax = RecordingAxes()

        draw_distribution_series(ax, data, series, color="C0", marker="o", linestyle="--")

        self.assertEqual(len(ax.plot_calls), 1)
        self.assertEqual(len(ax.errorbar_calls), 1)
        step_args, step_kwargs = ax.plot_calls[0]
        errorbar_args, errorbar_kwargs = ax.errorbar_calls[0]
        np.testing.assert_allclose(step_args[0], [0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
        np.testing.assert_allclose(step_args[1], [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0])
        self.assertEqual(step_kwargs["linestyle"], "--")
        self.assertNotIn("xerr", errorbar_kwargs)
        np.testing.assert_allclose(errorbar_kwargs["yerr"], series.errors)
        np.testing.assert_allclose(errorbar_args[0], data.bin_centers)
        np.testing.assert_allclose(errorbar_args[1], series.values)

    def test_distribution_series_style_keeps_sm_black_and_rotates_parameter_linestyles(self):
        color_cycle = ["C0", "C1", "C2"]
        point_one = DistributionSeries(
            label="c3=1",
            values=np.asarray([]),
            errors=np.asarray([]),
            parameter_values={"c3": 1.0},
        )
        point_two = DistributionSeries(
            label="c3=2",
            values=np.asarray([]),
            errors=np.asarray([]),
            parameter_values={"c3": 2.0},
        )
        sm = DistributionSeries(
            label="SM",
            values=np.asarray([]),
            errors=np.asarray([]),
            parameter_values={"c3": 0.0},
        )

        self.assertEqual(distribution_series_style(point_one, 0, color_cycle), ("C0", "-"))
        self.assertEqual(distribution_series_style(point_two, 1, color_cycle), ("C1", "--"))
        self.assertEqual(distribution_series_style(sm, 2, color_cycle), ("black", "-"))

    def test_parameter_points_default_unspecified_couplings_to_sm(self):
        config = load_config(ROOT / "configs" / "gg_tthhh_restricted5_ct_ct2_c3_validation.toml")

        points = parse_parameter_points(config, ["ct=0.5,c3=-0.25"])

        self.assertEqual(points, [{"ct": 0.5, "c3": -0.25}])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_distribution_command_writes_publication_style_plot(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        points = generate_scan_points(config)
        values = [point.values for point in points]
        observable = config.observables[0]
        fit_records = []
        for bin_index in range(len(observable.bins) - 1):
            scale = float(bin_index + 1)
            y = np.asarray([scale * (10.0 + 2.0 * c3 - 0.1 * d4) for c3, d4 in values])
            yerr = np.ones(len(values)) * 0.01
            fit_records.append(fit_values(config, values, y, yerr, f"{observable.name}:bin{bin_index}").to_dict(config))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fit_json = tmp_path / "hist_fit.json"
            fit_json.write_text(json.dumps({"fits": fit_records}), encoding="utf-8")
            output = tmp_path / "distribution.png"

            with patch.dict(os.environ, {"MPLCONFIGDIR": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path)}):
                status = main(
                    [
                        "distribution",
                        str(ROOT / "configs" / "gg_hhh_c3d4.toml"),
                        "--input",
                        str(fit_json),
                        "--observable",
                        observable.name,
                        "--point",
                        "c3=1,d4=2",
                        "--point",
                        "c3=-1",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

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
            self.assertIn("Inferred cross-section polynomial powers:", report)
            self.assertIn("  c3*d4", report)
            self.assertIn("  c3^2", report)
            self.assertTrue(report.rstrip().endswith("WARNING: CHECK inferred fit terms before using them."))

    def test_infers_terms_from_loop_induced_helas_ampl_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            process_dir = Path(tmpdir) / "proc"
            model_dir = process_dir / "bin" / "internal" / "ufomodel"
            matrix_dir = process_dir / "SubProcesses" / "PV0_0_1_gg_hh"
            model_dir.mkdir(parents=True)
            matrix_dir.mkdir(parents=True)
            (model_dir / "couplings.py").write_text(
                """
GC_30 = Coupling(name = 'GC_30',
                 value = '-6*complex(0,1)*lam*v*(1 + c3)',
                 order = {'QED':1})
""",
                encoding="utf-8",
            )
            (matrix_dir / "helas_calls_ampb_1.f").write_text(
                """
      SUBROUTINE ML5_0_0_1_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
      CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
      CALL SSS1_1(W(1,3),W(1,4),GC_30,MDL_MH,MDL_WH,W(1,5))
      CALL VVS1_0(W(1,1),W(1,2),W(1,5),R2_GGHB,AMPL(1,2))
      END
""",
                encoding="utf-8",
            )

            result = infer_terms_from_process_dir(process_dir, ("c3",))

            self.assertEqual(result.amplitude_terms, ((0,), (1,)))
            self.assertEqual(result.cross_section_terms, ((0,), (1,), (2,)))
            self.assertEqual(result.amplitude_support_counts[((0,), (1,))], 1)

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
            self.assertIn("Inferred cross-section polynomial powers:", output)
            self.assertIn("  c3*d4", output)
            self.assertIn("WARNING: Updated [fit].terms", output)
            self.assertIn("WARNING: Subsequent `fit` commands will use these inferred terms", output)
            self.assertIn("WARNING: CHECK the inferred terms before using them for production fits.", output)

            updated = load_config(config_path)
            self.assertEqual(
                updated.fit.terms,
                ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)),
            )

    def test_infer_terms_command_can_print_without_updating_config(self):
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
                status = main(["infer-terms", str(config_path), "--no-update-config"])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("WARNING: CHECK inferred fit terms before using them.", output)
            self.assertNotIn("Updated [fit].terms", output)
            unchanged = load_config(config_path)
            self.assertEqual(unchanged.fit.terms, ((0, 0),))

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
