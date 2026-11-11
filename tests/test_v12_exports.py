"""v1.2 包级导出锁定：三个新模块的公共 API 必须可从 fem 顶层导入。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fem


def test_v12_public_api_is_exported():
    required = (
        "solve_nonlinear_transient", "conduction_and_tangent", "energy_ledger",
        "energy_from_residual", "NonlinearTransientResult",
        "GeometryCache", "build_geometry_cache", "conduction_matrix",
        "TwoStageNetwork", "build_two_stage", "fit_tau_from_swing",
        "two_stage_swing", "diffusion_time_constant",
        "solve_radiative_steady", "solve_linearized_radiative_steady",
        "radiative_flux", "radiation_boundary", "effective_layer_k",
        "contact_rise_analytic", "case_with_contact", "STEFAN_BOLTZMANN",
    )
    for name in required:
        assert hasattr(fem, name), f"fem 顶层缺少 {name}"
        assert name in fem.__all__, f"__all__ 缺少 {name}"
