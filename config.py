"""All experimental settings; no region switches in simulation or plotting code."""

from dataclasses import asdict, dataclass
import math


REGIONS = {
    'Japan': {'place': 'Tokyo 23 Wards, Japan', 'epicenter': (35.6901, 139.7004)},
    'Hong Kong': {'place': 'Hong Kong', 'epicenter': (22.342, 114.179)},
    'Demo': {'place': None, 'epicenter': (35.0, 139.0)},
}


@dataclass(frozen=True)
class ExperimentConfig:
    region: str = 'Japan'
    n_agents: int = 1000
    seeds: tuple = tuple(range(32, 42))
    stress_levels: tuple = (3.5, 6.0, 8.5)
    severity_levels: tuple = (0.0, 3.0, 6.0)
    radius_km: float = 4.5
    epicenter: tuple | None = None
    closure_intensity: float | None = None
    threshold_low: float = 3.0
    threshold_high: float = 9.0
    logistic_slope: float = 2.0
    gamma: float = 0.5
    temperature: float = 0.05
    time_reference_sec: float = 900.0
    size_reference: float = 1000.0
    friction: float = 0.1
    exposure: float = 4.0
    requires_core_fraction: float = 0.2
    service_rate_multiplier: float = 1.0
    departure_window_min: float = 120.0
    access_target_min: float = 60.0
    bootstrap_samples: int = 2000
    analysis_seed: int = 2026
    recommendation_min_gain_min: float = 5.0
    recommendation_max_detour_min: float = 10.0
    recommendation_compliance: float = 0.8

    def __post_init__(self):
        for name in ('seeds', 'stress_levels', 'severity_levels'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.epicenter is not None:
            object.__setattr__(self, 'epicenter', tuple(self.epicenter))
        if self.region not in REGIONS:
            raise ValueError(f"Unknown region: {self.region}")
        for name in ('n_agents', 'bootstrap_samples'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be a non-empty list without duplicates")
        for seed in (*self.seeds, self.analysis_seed):
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("Seeds must be non-negative integers")
        for name in ('stress_levels', 'severity_levels'):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or not all(math.isfinite(x) and x >= 0 for x in values):
                raise ValueError(f"{name} must contain distinct finite non-negative values")
        if max(self.severity_levels) > 20:
            raise ValueError("Severity >20 is outside the numerical model domain")
        for name in ('radius_km', 'logistic_slope', 'time_reference_sec', 'size_reference',
                     'departure_window_min', 'access_target_min', 'service_rate_multiplier'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ('gamma', 'temperature', 'threshold_low', 'threshold_high', 'exposure',
                     'recommendation_min_gain_min', 'recommendation_max_detour_min'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.threshold_high <= self.threshold_low:
            raise ValueError("threshold_high must exceed threshold_low")
        if not math.isfinite(self.friction) or not 0 <= self.friction <= 1:
            raise ValueError("Friction must be in [0,1]")
        if not math.isfinite(self.requires_core_fraction) or not 0 <= self.requires_core_fraction <= 1:
            raise ValueError("requires_core_fraction must be in [0,1]")
        if not math.isfinite(self.recommendation_compliance) or not 0 <= self.recommendation_compliance <= 1:
            raise ValueError('Recommendation compliance must be in [0,1]')
        if self.closure_intensity is not None and not 0 < self.closure_intensity <= 1:
            raise ValueError("Closure threshold must be in (0,1]")
        lat, lon = self.hazard_center
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Epicenter must be valid latitude/longitude")

    @property
    def hazard_center(self):
        return self.epicenter if self.epicenter is not None else REGIONS[self.region]['epicenter']

    def to_dict(self):
        return asdict(self)

    @property
    def scenarios(self):
        return (
            ('Baseline', 1.0, 1.0, False),
            ('Full', self.friction, self.exposure, False),
            ('Friction', self.friction, 1.0, False),
            ('Exposure', 1.0, self.exposure, False),
            ('Fastest', 1.0, 1.0, True),
            ('Adaptive', 1.0, 1.0, False),
        )
