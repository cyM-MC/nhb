"""Aggregate the complete two-region OSM grid into publication export files."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import bootstrap_mean


def main():
    output = Path('results/osm_paper')
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    provenance = []
    for region in ('hk', 'japan'):
        for demand in (200, 1000, 2000):
            path = Path(f'results/osm_{region}_{demand}')
            frame = pd.read_csv(path / 'run_summary.csv')
            frame['demand'] = demand
            frames.append(frame)
            manifest = json.loads((path / 'manifest.json').read_text())
            provenance.append({'directory': str(path), 'graph_sha256': manifest['graph_sha256'],
                               'source_sha256': manifest['source_sha256']})
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(output / 'all_run_summary.csv', index=False)
    rows = []
    rng = np.random.default_rng(2026)
    for keys, group in summary.groupby(['region', 'demand', 'severity', 'stress', 'subgroup']):
        for metric in ('mean_access_min', 'mean_travel_min', 'mean_wait_min', 'p90_access_min',
                       'within_target_share', 'unreachable_share'):
            wide = group.pivot(index='seed', columns='scenario', values=metric)
            for scenario in ('Full', 'Fastest', 'Adaptive'):
                delta, lo, hi, n = bootstrap_mean(wide[scenario] - wide.Baseline, rng, 5000)
                rows.append(dict(zip(['region', 'demand', 'severity', 'stress', 'subgroup'], keys),
                                 scenario=scenario, metric=metric, baseline=wide.Baseline.mean(),
                                 intervention=wide[scenario].mean(), delta=delta,
                                 ci_lower=lo, ci_upper=hi, valid_seeds=n))
    effects = pd.DataFrame(rows)
    effects.to_csv(output / 'main_effects.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    colors = {'Full': '#b34d51', 'Fastest': '#666666', 'Adaptive': '#168b89'}
    for r, region in enumerate(('Hong Kong', 'Japan')):
        for c, demand in enumerate((200, 1000, 2000)):
            ax = axes[r, c]
            for scenario, color in colors.items():
                d = effects.loc[(effects.region == region) & (effects.demand == demand)
                                & (effects.subgroup == 'All') & (effects.metric == 'mean_access_min')
                                & (effects.scenario == scenario)].sort_values('severity')
                ax.plot(d.severity, d.delta, marker='o', label=scenario, color=color)
                ax.fill_between(d.severity, d.ci_lower, d.ci_upper, alpha=.15, color=color)
            ax.axhline(0, color='#999999', linewidth=.7)
            ax.set_title(f'{region}: {demand} patients / 120 min', fontsize=10)
            ax.set_xlabel('Hypothetical disaster severity')
            ax.set_ylabel('Access time difference (min)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.tick_params(axis='x', labelbottom=True)
    fig.suptitle('Intervention minus baseline: seed-paired mean access time', fontsize=12)
    fig.tight_layout()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'access_effects.{suffix}', dpi=300)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
