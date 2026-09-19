#!/usr/bin/env python3
"""100,000-run random-label permutation test for temporal local-support AUC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from lr_utils import (
    adjacency_from_edges, all_pairs, baseline_periodic_edges, load_edges,
    pair_scores, periodic_matrix, support_matrix, validate_temporal_edges,
)


def run(edges2025: str, edges2026: str, permutations: int = 100000,
        seed: int = 20260916, out_dir: str = 'results/permutation') -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    e25 = load_edges(edges2025)
    e26 = load_edges(edges2026)
    new_edges = validate_temporal_edges(e25, e26)

    A = adjacency_from_edges(e25)
    S = periodic_matrix(baseline_periodic_edges())
    M = support_matrix(A, S)

    nonedges = [p for p in all_pairs() if p not in e25]
    y = np.asarray([1 if p in new_edges else 0 for p in nonedges], dtype=np.int8)
    scores = pair_scores(M, nonedges)
    observed_auc = float(roc_auc_score(y, scores))

    # For a binary label with fixed number of positives, AUC can be computed from
    # the sum of average ranks. This makes 100,000 permutations fast and exact
    # with respect to ties in M.
    ranks = rankdata(scores, method='average')
    n = len(scores)
    n1 = int(y.sum())
    n0 = n - n1

    rng = np.random.default_rng(seed)
    aucs = np.empty(permutations, dtype=float)
    for b in range(permutations):
        pos = rng.choice(n, size=n1, replace=False)
        rank_sum = ranks[pos].sum()
        aucs[b] = (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    exceed = int(np.sum(aucs >= observed_auc))
    plus_one_p = float((exceed + 1) / (permutations + 1))
    result = {
        'candidate_nonedges': n,
        'later_recorded_positives': n1,
        'observed_auc': observed_auc,
        'permutations': permutations,
        'seed': seed,
        'null_mean': float(aucs.mean()),
        'null_sd': float(aucs.std(ddof=1)),
        'null_q999': float(np.quantile(aucs, 0.999)),
        'null_max': float(aucs.max()),
        'permutations_ge_observed': exceed,
        'plus_one_p': plus_one_p,
    }
    (out / 'permutation_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame({'permuted_auc': aucs}).to_csv(out / 'permutation_auc_values.csv', index=False)

    for k, v in result.items():
        print(f'{k}: {v}')
    print('\nInterpretation: this is a random-label null among 2025 candidate nonedges;')
    print('it is not a degree-preserving or node-dependence-preserving permutation.')
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edges2025', default='2-1946.txt')
    ap.add_argument('--edges2026', default='2-2006.txt')
    ap.add_argument('--permutations', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--out', default='results/permutation')
    args = ap.parse_args()
    run(args.edges2025, args.edges2026, args.permutations, args.seed, args.out)


if __name__ == '__main__':
    main()
