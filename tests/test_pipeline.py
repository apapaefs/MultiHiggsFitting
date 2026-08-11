from __future__ import annotations

import contextlib
import csv
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
    write_process_card,
)
from multihiggs.results import event_count, read_xsec
from multihiggs.term_inference import (
    format_inferred_terms,
    infer_terms_from_process_dir,
    project_amplitude_terms,
)
from multihiggs.term_maps import resolve_term_map


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


def write_mheft_process_fixture(process_dir: Path) -> None:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    matrix_dir = process_dir / "SubProcesses" / "P1_fixture"
    model_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    (model_dir / "couplings.py").write_text(
        """
GC_SM = Coupling(name = 'GC_SM',
                 value = 'yt',
                 order = {'QED':1})
GC_CT1 = Coupling(name = 'GC_CT1',
                  value = 'CT1',
                  order = {'MHEFT':1})
GC_D3 = Coupling(name = 'GC_D3',
                 value = 'D3',
                 order = {'MHEFT':1})
""",
        encoding="utf-8",
    )
    (matrix_dir / "matrix1_orig.f").write_text(
        """
      SUBROUTINE MATRIX1()
      CALL SSS1_0(W(1,1),W(1,2),GC_SM,AMP(1))
      CALL SSS1_0(W(1,1),W(1,2),GC_CT1,AMP(2))
      CALL SSS1_0(W(1,1),W(1,2),GC_D3,AMP(3))
      END
""",
        encoding="utf-8",
    )


def write_split_mheft_process_fixture(process_dir: Path) -> None:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    matrix_dir = process_dir / "SubProcesses" / "P1_fixture"
    model_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    (model_dir / "couplings.py").write_text(
        """
GC_OTHER = Coupling(name = 'GC_OTHER',
                    value = 'yt',
                    order = {'QED':1})
GC_30 = Coupling(name = 'GC_30',
                 value = '-6*complex(0,1)*lam*v',
                 order = {'QED':1})
GC_HHH_MHEFT = Coupling(name = 'GC_HHH_MHEFT',
                        value = '-6*D3*complex(0,1)*lam*v',
                        order = {'MHEFT':1,'QED':0})
""",
        encoding="utf-8",
    )
    (matrix_dir / "matrix1_orig.f").write_text(
        """
      SUBROUTINE MATRIX1()
      CALL SSS1_0(W(1,1),W(1,2),GC_OTHER,AMP(1))
      CALL SSS1_0(W(1,1),W(1,2),GC_30,AMP(2))
      CALL SSS1_0(W(1,1),W(1,2),GC_HHH_MHEFT,AMP(3))
      END
""",
        encoding="utf-8",
    )


def write_high_order_mheft_process_fixture(process_dir: Path) -> None:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    matrix_dir = process_dir / "SubProcesses" / "P1_fixture"
    model_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    (model_dir / "couplings.py").write_text(
        """
GC_SM = Coupling(name = 'GC_SM',
                 value = 'yt',
                 order = {'QED':1})
GC_D3_CUBED = Coupling(name = 'GC_D3_CUBED',
                       value = 'D3**3',
                       order = {'MHEFT':3})
""",
        encoding="utf-8",
    )
    (matrix_dir / "matrix1_orig.f").write_text(
        """
      SUBROUTINE MATRIX1()
      CALL SSS1_0(W(1,1),W(1,2),GC_SM,AMP(1))
      CALL SSS1_0(W(1,1),W(1,2),GC_D3_CUBED,AMP(2))
      END
""",
        encoding="utf-8",
    )


def write_madloop_tth_loop_fixture(process_dir: Path) -> None:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    matrix_dir = process_dir / "SubProcesses" / "PV0_0_1_gg_hhh"
    model_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    (model_dir / "couplings.py").write_text(
        """
GC_5 = Coupling(name = 'GC_5',
                value = 'complex(0,1)*G',
                order = {'QCD':1})
GC_37 = Coupling(name = 'GC_37',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                 order = {'QED':1})
GC_TTH_MHEFT = Coupling(name = 'GC_TTH_MHEFT',
                        value = '-CT1*((complex(0,1)*yt)/cmath.sqrt(2))',
                        order = {'MHEFT':1,'QED':0})
""",
        encoding="utf-8",
    )
    (matrix_dir / "helas_calls_ampb_1.f").write_text(
        """
      SUBROUTINE ML5_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
      CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
      CALL SXXXXX(P(0,5),+1*IC(5),W(1,5))
      END
""",
        encoding="utf-8",
    )
    (matrix_dir / "coef_construction_1.f").write_text(
        """
      SUBROUTINE ML5_COEF_CONSTRUCTION_1(P,NHEL,H,IC)
      CALL FFV1L1_2(PL(0,0),W(1,1),GC_5,MDL_MT,ZERO,PL(0,1),COEFS)
      CALL ML5_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))
      CALL FFS1L1_2(PL(0,1),W(1,3),GC_37,MDL_MT,MDL_WT,PL(0,2),COEFS)
      CALL ML5_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,2))
      CALL FFS1L1_2(PL(0,2),W(1,4),GC_37,MDL_MT,MDL_WT,PL(0,3),COEFS)
      CALL ML5_UPDATE_WL_2_1(WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3))
      CALL FFS1L1_2(PL(0,3),W(1,5),GC_37,MDL_MT,MDL_WT,PL(0,4),COEFS)
      CALL ML5_UPDATE_WL_3_1(WL(1,0,1,3),4,COEFS,4,4,WL(1,0,1,4))
      CALL ML5_CREATE_LOOP_COEFS(WL(1,0,1,4),5,4,1,1,1,1)
      CALL FFS1L1_2(PL(0,1),W(1,3),GC_TTH_MHEFT,MDL_MT,MDL_WT,PL(0,5),COEFS)
      CALL ML5_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,5))
      CALL FFS1L1_2(PL(0,5),W(1,4),GC_TTH_MHEFT,MDL_MT,MDL_WT,PL(0,6),COEFS)
      CALL ML5_UPDATE_WL_2_1(WL(1,0,1,5),4,COEFS,4,4,WL(1,0,1,6))
      CALL FFS1L1_2(PL(0,6),W(1,5),GC_TTH_MHEFT,MDL_MT,MDL_WT,PL(0,7),COEFS)
      CALL ML5_UPDATE_WL_3_1(WL(1,0,1,6),4,COEFS,4,4,WL(1,0,1,7))
      CALL ML5_CREATE_LOOP_COEFS(WL(1,0,1,7),5,4,2,1,1,2)
      END
""",
        encoding="utf-8",
    )


def write_sm_like_hhh_madloop_fixture(process_dir: Path) -> None:
    model_dir = process_dir / "bin" / "internal" / "ufomodel"
    matrix_dir = process_dir / "SubProcesses" / "PV0_0_1_gg_hhh"
    model_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    (model_dir / "couplings.py").write_text(
        """
GC_5 = Coupling(name = 'GC_5',
                value = 'complex(0,1)*G',
                order = {'QCD':1})
GC_30 = Coupling(name = 'GC_30',
                 value = '-6*complex(0,1)*lam*v',
                 order = {'QED':1})
GC_HHH_MHEFT = Coupling(name = 'GC_HHH_MHEFT',
                        value = '-6*D3*complex(0,1)*lam*v',
                        order = {'MHEFT':1,'QED':0})
GC_HHHH = Coupling(name = 'GC_HHHH',
                  value = '-6*complex(0,1)*lam',
                  order = {'QED':2})
GC_HHHH_MHEFT = Coupling(name = 'GC_HHHH_MHEFT',
                        value = '-6*D4*complex(0,1)*lam',
                        order = {'MHEFT':1,'QED':0})
GC_37 = Coupling(name = 'GC_37',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                 order = {'QED':1})
GC_TTH_MHEFT = Coupling(name = 'GC_TTH_MHEFT',
                        value = '-CT1*((complex(0,1)*yt)/cmath.sqrt(2))',
                        order = {'MHEFT':1,'QED':0})
""",
        encoding="utf-8",
    )
    (matrix_dir / "helas_calls_ampb_1.f").write_text(
        """
      SUBROUTINE ML5_HELAS_CALLS_AMPB_1(P,NHEL,H,IC)
      CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
      CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
      CALL SXXXXX(P(0,5),+1*IC(5),W(1,5))
      CALL SSS1_1(W(1,3),W(1,4),GC_30,MDL_MH,MDL_WH,W(1,6))
      CALL SSS1_1(W(1,5),W(1,6),GC_30,MDL_MH,MDL_WH,W(1,7))
      CALL SSSS1_1(W(1,3),W(1,4),W(1,5),GC_HHHH,MDL_MH,MDL_WH,W(1,8))
      CALL SSS1_1(W(1,3),W(1,4),GC_HHH_MHEFT,MDL_MH,MDL_WH,W(1,9))
      CALL SSSS1_1(W(1,3),W(1,4),W(1,5),GC_HHHH_MHEFT,MDL_MH,MDL_WH,W(1,10))
      END
""",
        encoding="utf-8",
    )
    (matrix_dir / "coef_construction_1.f").write_text(
        """
      SUBROUTINE ML5_COEF_CONSTRUCTION_1(P,NHEL,H,IC)
      CALL FFV1L1_2(PL(0,0),W(1,3),GC_5,MDL_MT,ZERO,PL(0,1),COEFS)
      CALL ML5_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))
      CALL FFS1L1_2(PL(0,1),W(1,3),GC_37,MDL_MT,MDL_WT,PL(0,2),COEFS)
      CALL ML5_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,2))
      CALL FFS1L1_2(PL(0,2),W(1,4),GC_37,MDL_MT,MDL_WT,PL(0,3),COEFS)
      CALL ML5_UPDATE_WL_2_1(WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3))
      CALL FFS1L1_2(PL(0,3),W(1,5),GC_37,MDL_MT,MDL_WT,PL(0,4),COEFS)
      CALL ML5_UPDATE_WL_3_1(WL(1,0,1,3),4,COEFS,4,4,WL(1,0,1,4))
      CALL ML5_CREATE_LOOP_COEFS(WL(1,0,1,4),5,4,1,1,1,1)
      CALL FFS1L1_2(PL(0,1),W(1,6),GC_TTH_MHEFT,MDL_MT,MDL_WT,PL(0,5),COEFS)
      CALL ML5_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,5))
      CALL FFS1L1_2(PL(0,5),W(1,5),GC_37,MDL_MT,MDL_WT,PL(0,6),COEFS)
      CALL ML5_UPDATE_WL_2_1(WL(1,0,1,5),4,COEFS,4,4,WL(1,0,1,6))
      CALL ML5_CREATE_LOOP_COEFS(WL(1,0,1,6),5,4,2,1,1,2)
      CALL FFS1L1_2(PL(0,1),W(1,7),GC_37,MDL_MT,MDL_WT,PL(0,7),COEFS)
      CALL ML5_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,7))
      CALL ML5_CREATE_LOOP_COEFS(WL(1,0,1,7),5,4,3,1,1,3)
      CALL FFS1L1_2(PL(0,1),W(1,8),GC_37,MDL_MT,MDL_WT,PL(0,8),COEFS)
      CALL ML5_UPDATE_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,8))
      CALL ML5_CREATE_LOOP_COEFS(WL(1,0,1,8),5,4,4,1,1,4)
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
        self.assertEqual(config.fit.basis, "physical_monomial")
        self.assertEqual(config.fit.term_map, "mheft_kappa")

    def test_hhhh_grid_matches_expected_size_and_terms(self):
        config = load_config(ROOT / "configs" / "gg_hhhh_c3d4.toml")
        points = generate_scan_points(config)
        self.assertEqual(len(points), 41)
        self.assertEqual(len(config.fit.terms), 15)
        self.assertEqual(config.fit.basis, "physical_monomial")
        self.assertEqual(config.fit.term_map, "mheft_kappa")

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
        self.assertEqual(config.fit.basis, "physical_monomial")
        self.assertEqual(config.fit.term_map, "mheft_kappa")
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
        self.assertEqual(config.fit.basis, "physical_monomial")
        self.assertEqual(config.fit.term_map, "mheft_kappa")
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
        self.assertEqual(config.fit.basis, "physical_monomial")
        self.assertEqual(config.fit.term_map, "mheft_kappa")
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

    def test_config_can_define_custom_term_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[process]
name = "fixture"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [[0]]

[term_maps.custom_top]
description = "Custom top modifier"

[[term_maps.custom_top.variables]]
source = "CT1"
name = "YT"
offset = 1.0
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertIn("custom_top", config.term_maps)
            term_map = config.term_maps["custom_top"]
            self.assertEqual(term_map.description, "Custom top modifier")
            self.assertEqual(term_map.variables[0].source, "CT1")
            self.assertEqual(term_map.variables[0].name, "YT")
            self.assertEqual(term_map.variables[0].offset, 1.0)

    def test_vv_kappa_couplings_select_vv_model_and_default_to_sm_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[process]
name = "fixture_vv"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "KZ"
parameter = "KZ"
fit_name = "KZ"
range = [0.5, 1.5]
points = 3

[[couplings]]
name = "KZZ"
parameter = "KZZ"
fit_name = "KZZ"
range = [0.5, 1.5]
points = 3

[[couplings]]
name = "KW"
parameter = "KW"
fit_name = "KW"
range = [0.5, 1.5]
points = 3

[[couplings]]
name = "KWW"
parameter = "KWW"
fit_name = "KWW"
range = [0.5, 1.5]
points = 3

[fit]
basis = "physical_monomial"
terms = [[0, 0, 0, 0]]
""",
                encoding="utf-8",
            )

            config = load_config(config_path)
            points = generate_scan_points(config)

            self.assertEqual(config.model, "heft_loop_sm_restricted5VV")
            self.assertEqual([coupling.sm_value for coupling in config.couplings], [1.0, 1.0, 1.0, 1.0])
            self.assertEqual(points[0].values, (1.0, 1.0, 1.0, 1.0))

    def test_vv_process_card_uses_vv_model_and_restricted_mheft_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            output = Path(tmpdir) / "process.mg5"
            config_path.write_text(
                """
[process]
name = "fixture_vv"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "KZ"
parameter = "KZ"
fit_name = "KZ"
range = [0.5, 1.5]
points = 3

[fit]
basis = "physical_monomial"
terms = [[0]]
""",
                encoding="utf-8",
            )

            config = load_config(config_path)
            write_process_card(config, output)

            text = output.read_text(encoding="utf-8")
            self.assertIn("import model heft_loop_sm_restricted5VV", text)
            self.assertIn("generate g g > h h [noborn=QCD MHEFT] MHEFT^2<=4", text)

    def test_mheft_kappa_term_map_includes_vv_kappas_as_physical_variables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """
[process]
name = "fixture_vv"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-0.5, 0.5]
points = 3

[[couplings]]
name = "KZ"
parameter = "KZ"
fit_name = "KZ"
range = [0.5, 1.5]
points = 3

[[couplings]]
name = "KZZ"
parameter = "KZZ"
fit_name = "KZZ"
range = [0.5, 1.5]
points = 3

[[couplings]]
name = "KW"
parameter = "KW"
fit_name = "KW"
range = [0.5, 1.5]
points = 3

[[couplings]]
name = "KWW"
parameter = "KWW"
fit_name = "KWW"
range = [0.5, 1.5]
points = 3

[fit]
basis = "physical_monomial"
terms = [[0, 0, 0, 0, 0]]
""",
                encoding="utf-8",
            )

            config = load_config(config_path)
            term_map = resolve_term_map(config, "mheft_kappa")

            self.assertEqual(term_map.names, ("KT", "KZ", "KZZ", "KW", "KWW"))
            self.assertEqual(term_map.offsets, (1.0, 0.0, 0.0, 0.0, 0.0))
            self.assertEqual(term_map.mapping_text(), "KT=1+CT1, KZ=KZ, KZZ=KZZ, KW=KW, KWW=KWW")

    def test_vv_ufo_sm_defaults_remain_external_under_restrictions(self):
        model_dir = ROOT / "models" / "heft_loop_sm_restricted5VV"
        parameters = (model_dir / "parameters.py").read_text(encoding="utf-8")

        for name in ("KZ", "KZZ", "KW", "KWW"):
            self.assertRegex(parameters, rf"(?s){name} = Parameter\(name = '{name}',.*?value = 1\.0,", msg=name)

        # MadGraph normalizes this sentinel to 1.0 while leaving the parameter
        # external.  A literal 1.0 in a restriction card would freeze it.
        vv_expected = {"KZ": 998, "KZZ": 999, "KW": 991, "KWW": 992}
        base_expected = {
            "CT1": (993, "1.3000e+00"),
            "CT2": (994, "1.4000e+00"),
            "CT3": (995, "1.5000e+00"),
            "D3": (996, "1.6000e+00"),
            "D4": (997, "1.7000e+00"),
        }
        restriction_cards = [
            *sorted(model_dir.glob("restrict*.dat")),
            *sorted(model_dir.glob(".restrict*.dat")),
        ]
        self.assertTrue(restriction_cards)
        for card in restriction_cards:
            restrict = card.read_text(encoding="utf-8")
            # Once BSMINPUTS is present, named restrictions must define the
            # complete external block or MadGraph rejects the import.
            for name, (lhacode, value) in base_expected.items():
                self.assertIn(f"    {lhacode} {value} # {name}", restrict, msg=f"{card.name}: {name}")
            for name, lhacode in vv_expected.items():
                self.assertIn(f"    {lhacode} 9.999999e-1 # {name}", restrict, msg=f"{card.name}: {name}")

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
        if config.fit.basis == "physical_monomial":
            self.assertEqual(config.fit.term_map, "mheft_kappa")
            self.assertGreater(len(config.fit.terms), 0)
            self.assertLessEqual(len(config.fit.terms), len(points))
            self.assertTrue(all(sum(term) <= 6 for term in config.fit.terms))
        else:
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
        self.assertTrue(all(len(term) == 3 for term in config.fit.terms))
        if config.fit.basis == "physical_monomial":
            self.assertEqual(config.fit.term_map, "mheft_kappa")
            self.assertTrue(all(sum(term) <= 6 for term in config.fit.terms))
        else:
            self.assertIn((0, 0, 0), config.fit.terms)

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
        if config.fit.basis == "physical_monomial":
            self.assertEqual(config.fit.term_map, "mheft_kappa")
            self.assertTrue(all(sum(term) <= 6 for term in config.fit.terms))
        else:
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

    def test_restricted5_process_card_inserts_mheft_noborn_and_squared_order_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "process.mg5"
            config = load_config(ROOT / "configs" / "gg_hhh_restricted5_ct_c3_d4_validation.toml")

            write_process_card(config, output)

            text = output.read_text(encoding="utf-8")
            self.assertIn("generate g g > h h h [noborn=QCD MHEFT] MHEFT^2<=6", text)
            self.assertNotIn("generate g g > h h h [noborn=QCD]", text)

    def test_restricted5_process_card_preserves_explicit_cap_and_normalizes_noborn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "process.mg5"
            config = load_config(ROOT / "configs" / "gg_hhh_restricted5_ct_c3_d4_validation.toml")
            config = replace(config, generate="g g > h h h [noborn=QCD] MHEFT^2<=4")

            write_process_card(config, output)

            text = output.read_text(encoding="utf-8")
            self.assertIn("generate g g > h h h [noborn=QCD MHEFT] MHEFT^2<=4", text)
            self.assertNotIn("MHEFT^2<=6", text)

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
        config = replace(
            config,
            fit=replace(
                config.fit,
                basis="chebyshev",
                term_map=None,
                terms=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)),
            ),
        )
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
        config = replace(
            config,
            fit=replace(
                config.fit,
                basis="chebyshev",
                term_map=None,
                terms=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)),
            ),
        )
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
        self.assertIn("Absolute physical-basis coefficients:", report)
        self.assertIn("Physical monomial coefficients in K3,K4:", report)
        self.assertIn("Physical polynomial coefficients in c3,d4:", report)

    def test_fit_polynomial_report_can_print_mheft_kappa_term_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                """
[process]
name = "fixture"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 3

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
]
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            xsecs = tmp_root / "xsecs.csv"
            with xsecs.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "run_name",
                        "run_number",
                        "ct",
                        "c3",
                        "xsec_pb",
                        "xerr_pb",
                        "event_count",
                        "event_file",
                    ],
                )
                writer.writeheader()
                for index, point in enumerate(generate_scan_points(config)):
                    ct, c3 = point.values
                    writer.writerow(
                        {
                            "run_name": f"run_{index}",
                            "run_number": "1",
                            "ct": ct,
                            "c3": c3,
                            "xsec_pb": (1.0 + ct) * (1.0 + c3),
                            "xerr_pb": 0.01,
                            "event_count": 100,
                            "event_file": tmp_root / f"run_{index}.lhe.gz",
                        }
                    )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "fit",
                    str(config_path),
                    "-i",
                    str(xsecs),
                    "-o",
                    str(tmp_root / "fit.json"),
                    "--print-polynomial",
                    "--term-map",
                    "mheft_kappa",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Physical polynomial coefficients in minimal term map mheft_kappa (KT,K3):", output)
            self.assertRegex(output, r"\(KT-1\)\*\(K3-1\)\s+1(?:\.0)?\b")
            self.assertNotIn("Expanded physical polynomial coefficients in term map mheft_kappa", output)

    def test_fit_polynomial_report_can_expand_mheft_kappa_term_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                """
[process]
name = "fixture"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 7

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 5

[fit]
basis = "chebyshev"
terms = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
]
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            xsecs = tmp_root / "xsecs.csv"
            with xsecs.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "run_name",
                        "run_number",
                        "ct",
                        "c3",
                        "xsec_pb",
                        "xerr_pb",
                        "event_count",
                        "event_file",
                    ],
                )
                writer.writeheader()
                for index, point in enumerate(generate_scan_points(config)):
                    ct, c3 = point.values
                    writer.writerow(
                        {
                            "run_name": f"run_{index}",
                            "run_number": "1",
                            "ct": ct,
                            "c3": c3,
                            "xsec_pb": (1.0 + ct) * (1.0 + c3),
                            "xerr_pb": 0.01,
                            "event_count": 100,
                            "event_file": tmp_root / f"run_{index}.lhe.gz",
                        }
                    )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "fit",
                    str(config_path),
                    "-i",
                    str(xsecs),
                    "-o",
                    str(tmp_root / "fit.json"),
                    "--print-polynomial",
                    "--term-map",
                    "mheft_kappa",
                    "--expand-term-map",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Expanded physical polynomial coefficients in term map mheft_kappa (KT,K3):", output)
            self.assertRegex(output, r"KT\*K3\s+1(?:\.0)?\b")

    def test_madevent_launch_block_disables_common_cuts(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")
        point = generate_scan_points(config)[0]
        block = "\n".join(build_launch_block(config, point))
        self.assertIn("set dsqrt_shat 0.0", block)
        self.assertIn("set ptheavy 0.0", block)
        self.assertIn("set pt_min_pdg {}", block)
        self.assertIn("set eta_max_pdg {}", block)
        self.assertIn("set mxx_min_pdg {}", block)
        self.assertIn(f"set nevents {config.scan.nevents}", block)

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
            self.assertIn("Number of polynomial terms: 6\nWARNING: CHECK", report)
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

    def test_infer_terms_command_can_print_mheft_kappa_term_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_mheft_process_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 3

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
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
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                    "--term-map",
                    "mheft_kappa",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Inferred cross-section minimal polynomial powers in term map mheft_kappa (KT,K3):", output)
            self.assertIn("Mapping: KT=1+CT1, K3=1+D3", output)
            self.assertIn("  (KT-1)*(K3-1)", output)
            self.assertNotIn("Inferred expanded cross-section polynomial powers in term map mheft_kappa", output)
            self.assertNotIn("Updated [fit].terms", output)

    def test_infer_terms_command_can_expand_mheft_kappa_term_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_mheft_process_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 3

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
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
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                    "--term-map",
                    "mheft_kappa",
                    "--expand-term-map",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Inferred expanded cross-section polynomial powers in term map mheft_kappa (KT,K3):", output)
            self.assertIn("  KT*K3", output)
            self.assertNotIn("Updated [fit].terms", output)

    def test_infer_terms_physical_basis_merges_restricted5_split_hhh_couplings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_split_mheft_process_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h h"
output = "proc"

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [[0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--term-map",
                    "mheft_kappa",
                    "--physical-basis",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Physical-basis fit variables: K3", output)
            self.assertIn("Inferred physical-basis cross-section [fit].terms:", output)
            self.assertIn("  K3^2", output)
            self.assertNotIn("  (K3-1)", output)
            updated = load_config(config_path)
            self.assertEqual(updated.fit.basis, "physical_monomial")
            self.assertEqual(updated.fit.term_map, "mheft_kappa")
            self.assertEqual(updated.fit.terms, ((0,), (1,), (2,)))

    def test_infer_terms_physical_basis_reads_madloop_top_yukawa_powers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_madloop_tth_loop_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 7

[fit]
basis = "chebyshev"
terms = [[0]]
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            source_names = tuple(coupling.parameter for coupling in config.couplings)
            term_map = resolve_term_map(config, "mheft_kappa", source_names=source_names)

            result = infer_terms_from_process_dir(
                process_dir,
                source_names,
                term_map=term_map,
                physical_basis=True,
                mheft_squared_order_cap=6,
            )

            self.assertIn((3,), result.amplitude_terms)
            self.assertIn((6,), result.cross_section_terms)
            self.assertEqual(result.coupling_names, ("KT",))
            self.assertEqual(result.coupling_supports["GC_37"], ((1,),))
            self.assertEqual(result.coupling_supports["GC_TTH_MHEFT"], ((1,),))

    def test_sm_like_hhh_amplitude_basis_preserves_non_sm_contact_terms(self):
        names = ("KT", "CT2", "CT3", "K3", "K4")
        amplitude_terms = {
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 0, 1),
            (3, 0, 0, 0, 0),
            (2, 0, 0, 1, 0),
            (1, 0, 0, 2, 0),
            (1, 0, 0, 0, 1),
            (0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0),
            (1, 1, 0, 0, 0),
        }

        projected = project_amplitude_terms(amplitude_terms, names, "sm_like_hhh")

        self.assertEqual(
            projected,
            {
                (3, 0, 0, 0, 0),
                (2, 0, 0, 1, 0),
                (1, 0, 0, 2, 0),
                (1, 0, 0, 0, 1),
                (0, 1, 0, 0, 0),
                (0, 0, 1, 0, 0),
                (1, 1, 0, 0, 0),
            },
        )

    def test_sm_like_hhh_amplitude_basis_allows_missing_k4(self):
        names = ("KT", "K3")
        amplitude_terms = {
            (0, 0),
            (0, 1),
            (1, 0),
            (3, 0),
            (2, 1),
            (1, 2),
        }

        projected = project_amplitude_terms(amplitude_terms, names, "sm_like_hhh")

        self.assertEqual(
            projected,
            {
                (1, 0),
                (3, 0),
                (2, 1),
                (1, 2),
            },
        )

    def test_sm_like_hhh_amplitude_basis_handles_tthh_subspace(self):
        names = ("KT", "CT2", "K3")
        amplitude_terms = {
            (0, 0, 0),
            (1, 0, 0),
            (0, 0, 1),
            (2, 0, 0),
            (1, 0, 1),
            (0, 1, 0),
        }

        projected = project_amplitude_terms(amplitude_terms, names, "sm_like_hhh")

        self.assertEqual(
            projected,
            {
                (2, 0, 0),
                (1, 0, 1),
                (0, 1, 0),
            },
        )

    def test_sm_like_amplitude_basis_handles_hhhh_subspace(self):
        names = ("KT", "K3", "K4", "CT2")
        amplitude_terms = {
            (0, 0, 0, 0),
            (2, 0, 0, 0),
            (1, 1, 0, 0),
            (3, 0, 0, 0),
            (2, 1, 0, 0),
            (1, 2, 0, 0),
            (1, 0, 1, 0),
            (4, 0, 0, 0),
            (3, 1, 0, 0),
            (2, 2, 0, 0),
            (1, 3, 0, 0),
            (2, 0, 1, 0),
            (1, 1, 1, 0),
            (0, 0, 0, 1),
        }

        projected = project_amplitude_terms(amplitude_terms, names, "sm_like")

        self.assertEqual(
            projected,
            {
                (4, 0, 0, 0),
                (3, 1, 0, 0),
                (2, 2, 0, 0),
                (1, 3, 0, 0),
                (2, 0, 1, 0),
                (1, 1, 1, 0),
                (0, 0, 0, 1),
            },
        )

    def test_sm_like_amplitude_basis_allows_missing_kt_for_c3d4_model(self):
        names = ("K3", "K4")
        amplitude_terms = {
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (3, 0),
            (1, 1),
        }

        projected = project_amplitude_terms(amplitude_terms, names, "sm_like")

        self.assertEqual(
            projected,
            {
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (3, 0),
                (1, 1),
            },
        )

    def test_infer_terms_sm_like_hhh_amplitude_basis_prints_compact_kappa_basis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_sm_like_hhh_madloop_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 7

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 7

[[couplings]]
name = "d4"
parameter = "D4"
fit_name = "d4"
range = [-1.0, 1.0]
points = 7

[fit]
basis = "chebyshev"
terms = [[0, 0, 0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                    "--term-map",
                    "mheft_kappa",
                    "--physical-basis",
                    "--amplitude-basis",
                    "sm_like_hhh",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Applied amplitude basis: sm_like_hhh", output)
            self.assertIn("  KT^6", output)
            self.assertIn("  KT^5*K3", output)
            self.assertIn("  KT^4*K3^2", output)
            self.assertIn("  KT^3*K3^3", output)
            self.assertIn("  KT^2*K3^4", output)
            self.assertIn("  KT^4*K4", output)
            self.assertIn("  KT^3*K3*K4", output)
            self.assertIn("  KT^2*K3^2*K4", output)
            self.assertIn("  KT^2*K4^2", output)
            self.assertIn("Number of polynomial terms: 9\nWARNING: CHECK", output)
            self.assertNotIn("  KT\n", output)
            self.assertNotIn("  K3\n", output)

    def test_builtin_mheft_kappa_map_matches_loop_c3d4_sources(self):
        config = load_config(ROOT / "configs" / "gg_hhh_c3d4.toml")

        term_map = resolve_term_map(config, "mheft_kappa", source_names=("c3", "d4"))

        self.assertEqual(term_map.names, ("K3", "K4"))
        self.assertEqual(term_map.mapping_text(), "K3=1+c3, K4=1+d4")

    def test_infer_terms_mheft_basis_sm_like_shorthand_handles_loop_c3d4_model(self):
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
generate = "g g > h h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "c3"
parameter = "c3"
fit_name = "k3"
range = [0.0, 2.0]
points = 3
fit_offset = 1.0

[[couplings]]
name = "d4"
parameter = "d4"
fit_name = "k4"
range = [0.0, 2.0]
points = 3
fit_offset = 1.0

[fit]
basis = "chebyshev"
terms = [[0, 0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                    "--mheft-basis",
                    "sm-like",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Applied amplitude basis: sm_like", output)
            self.assertIn("Physical-basis fit variables: K3,K4", output)
            self.assertIn("Mapping: K3=1+c3, K4=1+d4", output)
            self.assertIn("GC_C3: [1, 0]", output)
            self.assertIn("GC_D4: [0, 1]", output)
            self.assertIn("  1", output)
            self.assertIn("  K3", output)
            self.assertIn("  K4", output)
            self.assertIn("  K3*K4", output)
            self.assertIn("Number of polynomial terms: 6", output)

    def test_infer_terms_mheft_basis_sm_like_shorthand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_sm_like_hhh_madloop_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 7

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 7

[[couplings]]
name = "d4"
parameter = "D4"
fit_name = "d4"
range = [-1.0, 1.0]
points = 7

[fit]
basis = "chebyshev"
terms = [[0, 0, 0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                    "--mheft-basis",
                    "sm-like",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Applied amplitude basis: sm_like", output)
            self.assertIn("Physical-basis fit variables: KT,K3,K4", output)
            self.assertIn("Number of polynomial terms: 9", output)

    def test_fit_can_infer_physical_sm_like_hhh_basis_without_updating_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_sm_like_hhh_madloop_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 7

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 5

[[couplings]]
name = "d4"
parameter = "D4"
fit_name = "d4"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [[0, 0, 0]]
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            xsecs = tmp_root / "xsecs.csv"
            with xsecs.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "run_name",
                        "run_number",
                        "ct",
                        "c3",
                        "d4",
                        "xsec_pb",
                        "xerr_pb",
                        "event_count",
                        "event_file",
                    ],
                )
                writer.writeheader()
                for index, point in enumerate(generate_scan_points(config)):
                    ct, c3, d4 = point.values
                    kt = 1.0 + ct
                    k3 = 1.0 + c3
                    k4 = 1.0 + d4
                    value = (
                        kt**2 * k4**2
                        + kt**4 * k4
                        + kt**3 * k3 * k4
                        + kt**2 * k3**2 * k4
                        + kt**6
                        + kt**5 * k3
                        + kt**4 * k3**2
                        + kt**3 * k3**3
                        + kt**2 * k3**4
                    )
                    writer.writerow(
                        {
                            "run_name": point.run_name(config),
                            "run_number": "1",
                            "ct": ct,
                            "c3": c3,
                            "d4": d4,
                            "xsec_pb": value,
                            "xerr_pb": 0.01,
                            "event_count": 100,
                            "event_file": tmp_root / f"run_{index}.lhe.gz",
                        }
                    )

            output_path = tmp_root / "fit.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "fit",
                    str(config_path),
                    "-i",
                    str(xsecs),
                    "-o",
                    str(output_path),
                    "--mheft-basis",
                    "sm-like",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Using inferred physical fit basis: 9 term(s)", output)
            self.assertIn("Applied amplitude basis: sm_like", output)
            self.assertIn("rank=9/9", output)
            fit_data = json.loads(output_path.read_text(encoding="utf-8"))
            fit_record = fit_data["fits"][0]
            self.assertEqual(fit_record["basis"], "physical_monomial")
            self.assertEqual(len(fit_record["terms"]), 9)
            self.assertEqual(load_config(config_path).fit.terms, ((0, 0, 0),))

    def test_infer_terms_physical_basis_simplifies_one_plus_coupling_ufo_expressions(self):
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
generate = "g g > h h h"
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

[term_maps.loop_kappa]
description = "Loop model kappa modifiers"

[[term_maps.loop_kappa.variables]]
source = "c3"
name = "K3"
offset = 1.0

[[term_maps.loop_kappa.variables]]
source = "d4"
name = "K4"
offset = 1.0
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                    "--term-map",
                    "loop_kappa",
                    "--physical-basis",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Physical-basis fit variables: K3,K4", output)
            self.assertIn("GC_C3: [1, 0]", output)
            self.assertIn("GC_D4: [0, 1]", output)
            self.assertNotIn("GC_C3: [0, 0], [1, 0]", output)
            self.assertIn("  K3*K4", output)

    def test_infer_terms_applies_restricted_hh_mheft_squared_order_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_high_order_mheft_process_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [[0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Applied MHEFT^2 order cap: <= 4", output)
            self.assertIn("  [3],", output)
            self.assertNotIn("  [6],", output)
            self.assertNotIn("D3^6", output)

    def test_infer_terms_applies_restricted_hhh_mheft_squared_order_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            mg5_path = tmp_root / "MG5"
            process_dir = mg5_path / "proc"
            write_high_order_mheft_process_fixture(process_dir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                f"""
[process]
name = "fixture"
mg5_path = "{mg5_path}"
model = "heft_loop_sm_restricted5"
generate = "g g > h h h [noborn=QCD]"
output = "proc"

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "chebyshev"
terms = [[0]]
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([
                    "infer-terms",
                    str(config_path),
                    "--no-update-config",
                ])

            self.assertEqual(status, 0)
            output = stdout.getvalue()
            self.assertIn("Applied MHEFT^2 order cap: <= 6", output)
            self.assertIn("  [6],", output)
            self.assertIn("  D3^6", output)

    def test_fit_physical_monomial_basis_uses_mapped_variables_as_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            config_path = tmp_root / "config.toml"
            config_path.write_text(
                """
[process]
name = "fixture"
mg5_path = "MG5"
model = "heft_loop_sm_restricted5"
generate = "g g > h h"
output = "proc"

[[couplings]]
name = "ct"
parameter = "CT1"
fit_name = "ct"
range = [-1.0, 1.0]
points = 3

[[couplings]]
name = "c3"
parameter = "D3"
fit_name = "c3"
range = [-1.0, 1.0]
points = 3

[fit]
basis = "physical_monomial"
term_map = "mheft_kappa"
terms = [
  [1, 1],
]
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            values = [point.values for point in generate_scan_points(config)]
            y = np.asarray([(1.0 + ct) * (1.0 + c3) for ct, c3 in values])
            yerr = np.ones(len(values)) * 0.01

            result = fit_values(config, values, y, yerr, "synthetic")
            report = format_polynomial_report(config, result)

            self.assertEqual(result.rank, 1)
            self.assertAlmostEqual(result.coefficients[0], 1.0)
            self.assertAlmostEqual(result.sigma_sm, 1.0)
            self.assertIn("Physical monomial coefficients in KT,K3:", report)
            self.assertRegex(report, r"KT\*K3\s+1(?:\.0)?\b")

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
