"""材料层与传热工况（全部 SI 单位；二维模型按单位厚度 1 m，功率用线密度 W/m）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    """从底面往上堆叠的一层均质材料。"""

    name: str
    thickness: float  # m
    k: float  # W/(m K) 导热系数
    rc: float  # J/(m^3 K) 体积热容 rho*c


@dataclass(frozen=True)
class ThermalCase:
    """芯片-散热结构瞬态热传导工况。

    几何：x in [0, width]，y in [0, height]，layers 从底到顶。
    边界：底面 y=0 对流 (h_conv, t_amb)；顶面与左右侧面绝热（自然边界）。
    热源：顶层（芯片层）中心区域 [x0, x1] 均匀体积热源；
          heat_source_width == width 时退化为严格一维问题。
    """

    width: float
    layers: tuple[Layer, ...]
    h_conv: float
    t_amb: float
    power_per_length: float  # W/m，单位厚度上的总功率
    heat_source_width: float
    nx: int
    ny_per_layer: int
    theta: float = 1.0  # 1.0 后向 Euler；0.5 Crank-Nicolson
    dt: float = 0.01  # s
    t_end: float = 10.0  # s
    pulse_period: float = 2.0  # s
    duty: float = 0.5

    @property
    def height(self) -> float:
        return sum(L.thickness for L in self.layers)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def y_interfaces(self) -> list[float]:
        ys = [0.0]
        for L in self.layers:
            ys.append(ys[-1] + L.thickness)
        return ys

    @property
    def heat_source_bounds(self) -> tuple[float, float]:
        x0 = 0.5 * (self.width - self.heat_source_width)
        return x0, x0 + self.heat_source_width

    @property
    def die(self) -> Layer:
        return self.layers[-1]

    # ---- 集总热阻（1D 串联，单位厚度） ----
    @property
    def r_cond_series(self) -> float:
        """各层导热热阻之和 (K m / W)。"""
        return sum(L.thickness / (L.k * self.width) for L in self.layers)

    @property
    def r_conv(self) -> float:
        """底面对流热阻 (K m / W)。"""
        return 1.0 / (self.h_conv * self.width)

    @property
    def r_total(self) -> float:
        """表面热源口径：全部热流穿过每一层的串联热阻 + 对流。"""
        return self.r_cond_series + self.r_conv

    @property
    def r_effective(self) -> float:
        """体积热源口径（与 FEM/Cauer 一致）的稳态有效热阻。

        芯片层内均匀体积热源时，层内热流从底面 q'' 线性降到顶面 0，
        故该层温升只有 q''*L/(2k)，等效热阻为 L/(2kW)；其余层仍为 L/(kW)。
        """
        r = sum(L.thickness / (L.k * self.width) for L in self.layers[:-1])
        r += self.die.thickness / (2.0 * self.die.k * self.width)
        return r + self.r_conv

    @property
    def t_junction_steady(self) -> float:
        """1D 稳态解析（体积热源口径）：顶面芯片结温。"""
        return self.t_amb + self.power_per_length * self.r_effective

    @property
    def t_junction_surface_source(self) -> float:
        """朴素串联热阻口径（热源集中于顶面）的结温，用于量化口径差。"""
        return self.t_amb + self.power_per_length * self.r_total

    @property
    def heat_capacity_per_length(self) -> float:
        """单位厚度总热容 (J/(m K))。"""
        return sum(L.rc * L.thickness * self.width for L in self.layers)

    @property
    def cauer_time_constant(self) -> float:
        """单节集总 Cauer 的一阶时间常数 tau = C_total * R_conv (s)。"""
        return self.heat_capacity_per_length * self.r_conv

    def with_params(self, **kw) -> "ThermalCase":
        from dataclasses import replace

        return replace(self, **kw)


def default_case() -> ThermalCase:
    """典型功率器件封装（Si die / TIM / Cu base），单位厚度二维模型。"""
    layers = (
        Layer("Cu_base", 1.0e-3, 400.0, 3.45e6),
        Layer("TIM", 0.2e-3, 5.0, 2.0e6),
        Layer("Si_die", 0.3e-3, 130.0, 1.65e6),
    )
    return ThermalCase(
        width=10.0e-3,
        layers=layers,
        h_conv=5000.0,
        t_amb=25.0,
        power_per_length=3000.0,
        heat_source_width=10.0e-3,  # 默认一维验证口径
        nx=80,
        ny_per_layer=8,
        theta=1.0,
        dt=0.01,
        t_end=10.0,
        pulse_period=2.0,
        duty=0.5,
    )
