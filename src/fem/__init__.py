"""N6 有限元瞬态热仿真：手写 Galerkin FEM + Cauer 热阻网络交叉验证。"""
from .materials import Layer, ThermalCase, default_case
from .mesh import StructuredMesh, build_mesh
from .assembly import FEMSystem, assemble, solve_steady
from .solver import (TransientResult, power_pulse, run_transient,
                     steady_field, periodic_steady_metrics,
                     peak_over_last_cycle, trough_over_last_cycle)
from .cauer import CauerNetwork, fine_network, lumped_network
from .validation import (analytic_1d_steady, analytic_1d_field, skfem_steady,
                         error_metrics, convergence_order, bi_numbers)
from .postprocess import (curve_error_full, thermal_metrics,
                          spreading_resistance, fit_first_order_tau, sweep_table)
from .config import load_config, case_from_config, default_config_path

__all__ = [
    "Layer", "ThermalCase", "default_case",
    "StructuredMesh", "build_mesh",
    "FEMSystem", "assemble", "solve_steady",
    "TransientResult", "power_pulse", "run_transient", "steady_field",
    "periodic_steady_metrics", "peak_over_last_cycle", "trough_over_last_cycle",
    "CauerNetwork", "fine_network", "lumped_network",
    "analytic_1d_steady", "analytic_1d_field", "skfem_steady",
    "error_metrics", "convergence_order", "bi_numbers",
    "curve_error_full", "thermal_metrics", "spreading_resistance",
    "fit_first_order_tau", "sweep_table",
    "load_config", "case_from_config", "default_config_path",
]
