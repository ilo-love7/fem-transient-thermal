"""v1.3 E17：多工况联合辨识的单测。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fem.config import case_from_config, default_config_path, load_config  # noqa: E402
from fem.multicondition import (Condition, condition_suite, joint_forward,  # noqa: E402
                                joint_fisher, periodic_phase_observations,
                                steady_observations)


@pytest.fixture(scope="module")
def base():
    return case_from_config(load_config(default_config_path()))


THETA = np.log(np.array([5.0, 5000.0]))


def test_steady_observations_scale_with_power(base):
    obs = steady_observations(THETA, base, (0.5, 1.0, 1.5), n=8)
    rises = obs - base.t_amb
    assert rises[1] == pytest.approx(2 * rises[0], rel=1e-9)
    assert rises[2] == pytest.approx(3 * rises[0], rel=1e-9)


def test_periodic_phases_in_one_cycle(base):
    ph = (0.25, 0.5, 0.75, 0.95)
    obs = periodic_phase_observations(THETA, base, 1.0, 0.5, 0.5, ph,
                                      n=8, n_steps_per_period=40, n_periods=8)
    assert obs.shape == (4,)
    assert np.all(np.isfinite(obs))
    # 导通相（0.25）温度高于关断相尾部（0.75, 0.95）
    assert obs[0] > obs[2] and obs[0] > obs[3]


def test_joint_forward_concatenation(base):
    conds = (Condition("s", power=1.0),
             Condition("p", power=1.0, period=0.5, phases=(0.25, 0.75)))
    f = joint_forward(THETA, base, conds, n=8)
    a = steady_observations(THETA, base, (1.0,), n=8)
    b = periodic_phase_observations(THETA, base, 1.0, 0.5, 0.5,
                                    (0.25, 0.75), n=8)
    np.testing.assert_allclose(f, np.concatenate([a, b]))


def test_steady_only_fim_is_rank_one(base):
    suite = condition_suite(base)
    fim = joint_fisher(THETA, base, suite["steady_only"], n=8)
    # 多功率稳态仍秩 1：功率维只缩放同一温度场，不带新方向信息
    assert fim["effective_rank"] == 1
    assert fim["condition_number"] > 1e8
    assert abs(fim["correlation"]) > 0.999


def test_multi_period_fim_breaks_rank_one(base):
    suite = condition_suite(base)
    steady = joint_fisher(THETA, base, suite["steady_only"], n=8)
    multi = joint_fisher(THETA, base, suite["multi_period"], n=8)
    assert steady["effective_rank"] == 1
    assert multi["effective_rank"] == 2
    assert multi["condition_number"] < steady["condition_number"] / 1e4


def test_multi_condition_reduces_parameter_correlation(base):
    suite = condition_suite(base)
    a = joint_fisher(THETA, base, suite["steady_only"], n=8)
    b = joint_fisher(THETA, base, suite["multi_period"], n=8)
    c = joint_fisher(THETA, base, suite["multi_power_duty"], n=8)
    # 稳态的参数相关=1（完全不可分）；多工况把相关性显著拉低，条件数下降
    assert abs(a["correlation"]) > 0.999
    assert abs(b["correlation"]) < 0.9
    assert c["condition_number"] < b["condition_number"] < a["condition_number"] / 1e100


def test_fisher_jacobian_shape_and_symmetry(base):
    suite = condition_suite(base)
    fim = joint_fisher(THETA, base, suite["single_period"], n=8)
    assert fim["jacobian"].shape[1] == 2
    assert fim["cov_theta"].shape == (2, 2)
    assert fim["eigenvalues"][0] >= 0
    assert fim["eigenvalues"][1] >= fim["eigenvalues"][0]
    assert fim["n_observations"] == fim["jacobian"].shape[0]


def test_condition_suite_has_four_regimes(base):
    s = condition_suite(base)
    assert set(s) == {"steady_only", "single_period", "multi_period",
                      "multi_power_duty"}
    assert all(isinstance(v, tuple) and v for v in s.values())
