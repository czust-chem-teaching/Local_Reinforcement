#!/usr/bin/env python3
"""Random edge-removal robustness experiment used in Section 4.1."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lr_utils import (
    adjacency_from_edges, all_pairs, baseline_periodic_edges, load_edges,
    periodic_matrix, support_matrix,
)


def run(edges2025: str, reps: int = 1000, seed: int = 20260916,
        out_dir: str = 'results/robustness') -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    edge_set = load_edges(edges2025)
    edge_pairs = sorted(edge_set)
    A = adjacency_from_edges(edge_set)
    S = periodic_matrix(baseline_periodic_edges())

    nonedges = [p for p in all_pairs() if p not in edge_set]
    edge_arr = np.asarray(edge_pairs, dtype=np.int16)
    non_arr = np.asarray(nonedges, dtype=np.int16)
    rng = np.random.default_rng(seed)

    trial_rows = []
    for removal_fraction in [0.05, 0.10, 0.20, 0.30]:
        n_remove = int(round(removal_fraction * len(edge_pairs)))
        for rep in range(reps):
            selected = rng.choice(len(edge_pairs), size=n_remove, replace=False)
            hidden = edge_arr[selected]

            Ar = A.copy()
            Ar[hidden[:, 0], hidden[:, 1]] = 0
            Ar[hidden[:, 1], hidden[:, 0]] = 0
            Mr = support_matrix(Ar, S)

            candidates = np.vstack([non_arr, hidden])
            y = np.concatenate([
                np.zeros(len(non_arr), dtype=np.int8),
                np.ones(len(hidden), dtype=np.int8),
            ])
            scores = Mr[candidates[:, 0], candidates[:, 1]].astype(float)
            auc = float(roc_auc_score(y, scores))
            cls = np.minimum(scores.astype(int), 3)

            prevalence = []
            for c in range(4):
                mask = cls == c
                prevalence.append(float(y[mask].mean()))

            trial_rows.append({
                'removal_fraction': removal_fraction,
                'removed_edges': n_remove,
                'replicate': rep + 1,
                'hidden_prev_M0_pct': 100 * prevalence[0],
                'hidden_prev_M1_pct': 100 * prevalence[1],
                'hidden_prev_M2_pct': 100 * prevalence[2],
                'hidden_prev_M3plus_pct': 100 * prevalence[3],
                'auc': auc,
                'strict_monotonic': bool(prevalence[0] < prevalence[1] < prevalence[2] < prevalence[3]),
            })

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(out / 'robustness_trials_2025.csv', index=False)

    summaries = []
    for frac, g in trials.groupby('removal_fraction', sort=True):
        row = {
            'removal_fraction': frac,
            'removed_edges': int(g['removed_edges'].iloc[0]),
            'auc_mean': g['auc'].mean(),
            'auc_sd': g['auc'].std(ddof=1),
            'strict_monotonic_pct': 100 * g['strict_monotonic'].mean(),
        }
        for label in ['M0', 'M1', 'M2', 'M3plus']:
            col = f'hidden_prev_{label}_pct'
            row[f'hidden_prev_{label}_mean_pct'] = g[col].mean()
            row[f'hidden_prev_{label}_sd_pct'] = g[col].std(ddof=1)
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'robustness_summary_2025.csv', index=False)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edges2025', default='2-1946.txt')
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--out', default='results/robustness')
    args = ap.parse_args()
    run(args.edges2025, args.reps, args.seed, args.out)


if __name__ == '__main__':
    main()
