"""Run-level outcomes and seed-paired Monte Carlo uncertainty (not patient p-values)."""

import numpy as np
import pandas as pd


RUN_KEYS = ['region', 'seed', 'severity', 'stress', 'scenario']
EFFECT_METRICS = (
    'unreachable_share', 'within_target_share', 'mean_access_min',
    'mean_travel_min', 'mean_wait_min', 'p90_access_min',
    'mean_excess_travel_min', 'core_share_reachable', 'demand_capacity_cv',
)


def facility_loads(decisions, facilities, config):
    """Include every facility, even those with zero demand."""
    loads = facilities[['name', 'facility_type', 'servers', 'service_minutes']].copy()
    counts = decisions['chosen_facility'].value_counts()
    loads['patients'] = loads['name'].map(counts).fillna(0).astype(int)
    throughput = loads['servers'] * 60 / loads['service_minutes']
    loads['demand_capacity_ratio'] = loads['patients'] / (throughput * config.departure_window_min / 60)
    waits = decisions.groupby('chosen_facility')['wait_min'].mean()
    loads['mean_wait_min'] = loads['name'].map(waits)
    return loads


def summarize_run(decisions, facilities, config):
    low_threshold = config.threshold_low + .25 * (config.threshold_high - config.threshold_low)
    subsets = {
        'All': decisions,
        'LowThreshold': decisions.loc[decisions['threshold'] <= low_threshold],
        'RequiresCore': decisions.loc[decisions['requires_core']],
        'ClinicEligible': decisions.loc[~decisions['requires_core']],
    }
    loads = facility_loads(decisions, facilities, config)
    rows = []
    for subgroup, df in subsets.items():
        reachable = df.loc[df['reachable']]
        row = {key: decisions.iloc[0][key] for key in RUN_KEYS}
        row.update(subgroup=subgroup, agents=len(df), reachable_agents=len(reachable),
                   unreachable_share=(~df['reachable']).mean(),
                   within_target_share=df['within_target'].mean(),
                   mean_access_min=reachable['access_min'].mean(),
                   p90_access_min=reachable['access_min'].quantile(.9),
                   mean_travel_min=reachable['travel_min'].mean(),
                   mean_wait_min=reachable['wait_min'].mean(),
                   mean_excess_travel_min=reachable['excess_travel_min'].mean(),
                   core_share_reachable=(reachable['facility_type'] == 'core').mean())
        total = loads['patients'].sum()
        ratios = loads['demand_capacity_ratio'].to_numpy()
        row['demand_capacity_cv'] = (
            ratios.std(ddof=0) / ratios.mean() if subgroup == 'All' and total > 0 else np.nan)
        row['load_hhi'] = (
            ((loads['patients'] / total)**2).sum() if subgroup == 'All' and total > 0 else np.nan)
        rows.append(row)
    return rows, loads


def bootstrap_mean(values, rng, samples):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return np.nan, np.nan, np.nan, 0
    if n == 1:
        return float(values[0]), np.nan, np.nan, 1
    # Batching bounds memory for large repetition counts.
    boot = np.empty(samples)
    for start in range(0, samples, 250):
        end = min(samples, start + 250)
        indices = rng.integers(0, n, size=(end - start, n))
        boot[start:end] = values[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [.025, .975])
    return float(values.mean()), float(lower), float(upper), n


def paired_effects(run_summary, config):
    """Treatment-baseline seed differences and factorial interaction.

    The uncertainty unit is the population seed conditional on one fixed road
    network and model settings. These are not empirical population intervals.
    """
    keys = ['region', 'severity', 'stress', 'subgroup']
    if run_summary.duplicated(keys + ['seed', 'scenario']).any():
        raise ValueError("Duplicate seed/scenario summaries")
    rng = np.random.default_rng(config.analysis_seed)
    rows = []
    scenarios = [s[0] for s in config.scenarios]
    for setting, group in run_summary.groupby(keys, sort=True):
        for metric in EFFECT_METRICS:
            wide = group.pivot(index='seed', columns='scenario', values=metric)
            seed_sets = group.groupby('scenario')['seed'].agg(lambda values: set(values))
            if set(seed_sets.index) != set(scenarios) or any(seeds != set(config.seeds) for seeds in seed_sets):
                raise ValueError("Seed pairing requires all scenarios for every seed")
            deltas = {name: wide[name] - wide['Baseline'] for name in scenarios if name != 'Baseline'}
            deltas['Interaction'] = wide['Full'] - wide['Friction'] - wide['Exposure'] + wide['Baseline']
            for comparison, values in deltas.items():
                mean, lower, upper, n = bootstrap_mean(values, rng, config.bootstrap_samples)
                rows.append(dict(zip(keys, setting), comparison=comparison, metric=metric,
                                 delta=mean, ci_lower=lower, ci_upper=upper, valid_seeds=n,
                                 requested_seeds=len(config.seeds)))
    return pd.DataFrame(rows)
