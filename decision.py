"""Dimensionless one-shot facility choice, independent of routing and queues."""

import numpy as np


def rationality_weight(thresholds, stress, slope=2.0):
    """Imposed logistic hypothesis, not a behavioural estimate from simulations."""
    thresholds = np.asarray(thresholds, dtype=float)
    stress = np.asarray(stress, dtype=float)
    if not np.isfinite(thresholds).all() or not np.isfinite(stress).all():
        raise ValueError("Stress and thresholds must be finite")
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("Logistic slope must be positive")
    z = np.clip(slope * (thresholds - stress), -700, 700)
    return 1.0 / (1.0 + np.exp(-z))


def choose_facilities(travel_times, facilities, thresholds, stress, requires_core,
                      config, friction=1.0, exposure=1.0, noise=None, fastest=False):
    """Return facility indices (-1 if inaccessible) and rationality weights.

    R = t_ref / (t_ref + t)
    H = prestige * (size / size_ref) * R**gamma * information_multiplier
    U = alpha * R + (1-alpha) * H + temperature * fixed_Gumbel_noise

    Reference scales are fixed across treatments. Common noise is supplied by
    the runner. Faster travel is a benchmark, not optimal total access time.
    """
    times = np.asarray(travel_times, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    requires_core = np.asarray(requires_core, dtype=bool)
    n = len(thresholds)
    if times.shape != (n, len(facilities)) or requires_core.shape != (n,):
        raise ValueError("Choice arrays have incompatible shapes")
    if not len(facilities):
        raise ValueError("At least one facility is required")
    if np.isnan(times).any() or (times < 0).any():
        raise ValueError("Travel times must be non-negative or +inf")
    if not all(np.isfinite(x) and x >= 0 for x in (friction, exposure)):
        raise ValueError("Information multipliers must be non-negative and finite")
    core = facilities['facility_type'].to_numpy() == 'core'
    eligible = np.isfinite(times) & (~requires_core[:, None] | core[None, :])
    alpha = np.ones(n) if fastest else rationality_weight(thresholds, stress, config.logistic_slope)
    if fastest:
        scores = -times
    else:
        rational = config.time_reference_sec / (config.time_reference_sec + times)
        attraction = (facilities['prestige'].to_numpy()
                      * facilities['attraction_size'].to_numpy() / config.size_reference)
        multiplier = np.where(core, friction, exposure)
        heuristic = attraction * rational**config.gamma * multiplier
        scores = alpha[:, None] * rational + (1 - alpha[:, None]) * heuristic
        if config.temperature > 0:
            if noise is None or np.shape(noise) != times.shape or not np.isfinite(noise).all():
                raise ValueError("Supply finite common noise for stochastic choices")
            scores = scores + config.temperature * noise
    scores = np.where(eligible, scores, -np.inf)
    # Input rows are sorted by name; exact ties have a fixed alphabetical rule.
    choices = scores.argmax(axis=1)
    choices[~eligible.any(axis=1)] = -1
    return choices, alpha
