"""N6 有限元瞬态热仿真（v1.2）：手写 Galerkin FEM + Cauer 网络 +
轴对称三维扩散修正 + 非线性物性 + 参数反辨识 +
k(T) 瞬态 Newton + 动态扩散 RC + 辐射/接触非线性边界。"""
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
from .axisym import (AxisymCase, AxisymMesh, AxisymSystem, axisym_from_case,
                     build_axisym_mesh, assemble_axisym, solve_axisym_steady,
                     axisym_junction, run_axisym_transient,
                     spreading_resistance_abs, semi_infinite_spreading)
from .spreading import (SpreadingClosure, fit_spreading_closure,
                        equivalent_ladder_case, add_static_spreading,
                        corrected_ladder)
from .nonlinear import (NonlinearResult, solve_nonlinear_steady,
                        rc_periodic_closed_form, theta_artifacts)
from .identification import (build_case, make_network, exact_step_response,
                             forward, synthesize, fit, fisher,
                             OBSERVABLES, STEP_TIMES, POWER_LEVELS)
from .nonlinear_transient import (
    NonlinearTransientResult, solve_nonlinear_transient, split_system,
    k_elements, conduction_and_tangent, nonlinear_residual,
    energy_ledger, energy_from_residual,
    GeometryCache, build_geometry_cache, conduction_matrix)
from .two_stage import (
    TwoStageNetwork, build_two_stage, diffusion_time_constant,
    diffusion_pole_closed_form, two_stage_swing, fit_tau_from_swing)
from .boundary_nonlinear import (
    STEFAN_BOLTZMANN, RadiativeResult, radiative_flux,
    radiative_conductance_secant, radiative_conductance_local,
    linearization_error, radiation_boundary, linearized_radiation_matrix,
    solve_radiative_steady, solve_linearized_radiative_steady,
    effective_layer_k, contact_rise_analytic, case_with_contact)

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
    "AxisymCase", "AxisymMesh", "AxisymSystem", "axisym_from_case",
    "build_axisym_mesh", "assemble_axisym", "solve_axisym_steady",
    "axisym_junction", "run_axisym_transient",
    "spreading_resistance_abs", "semi_infinite_spreading",
    "SpreadingClosure", "fit_spreading_closure", "equivalent_ladder_case",
    "add_static_spreading", "corrected_ladder",
    "NonlinearResult", "solve_nonlinear_steady", "rc_periodic_closed_form",
    "theta_artifacts",
    "build_case", "make_network", "exact_step_response", "forward",
    "synthesize", "fit", "fisher", "OBSERVABLES", "STEP_TIMES", "POWER_LEVELS",
    "NonlinearTransientResult", "solve_nonlinear_transient", "split_system",
    "k_elements", "conduction_and_tangent", "nonlinear_residual",
    "energy_ledger", "energy_from_residual",
    "GeometryCache", "build_geometry_cache", "conduction_matrix",
    "TwoStageNetwork", "build_two_stage", "diffusion_time_constant",
    "diffusion_pole_closed_form", "two_stage_swing", "fit_tau_from_swing",
    "STEFAN_BOLTZMANN", "RadiativeResult", "radiative_flux",
    "radiative_conductance_secant", "radiative_conductance_local",
    "linearization_error", "radiation_boundary", "linearized_radiation_matrix",
    "solve_radiative_steady", "solve_linearized_radiative_steady",
    "effective_layer_k", "contact_rise_analytic", "case_with_contact",
]
