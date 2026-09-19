#!/usr/bin/env python3
"""Sensitivity of static and temporal AUC to alternative definitions of S."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from lr_utils import (
    adjacency_from_edges, all_pairs, load_edges, pair_scores,
    periodic_edge_variants, periodic_matrix, support_matrix,
    validate_temporal_edges,
)


def run(edges2025: str, edges2026: str,
        out_dir: str = 'results/s_sensitivity') -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    e25 = load_edges(edges2025)
    e26 = load_edges(edges2026)
    new_edges = validate_temporal_edges(e25, e26)
    A = adjacency_from_edges(e25)

    pairs = all_pairs()
    nonedges = [p for p in pairs if p not in e25]
    y_static = np.asarray([1 if p in e25 else 0 for p in pairs], dtype=np.int8)
    y_temp = np.asarray([1 if p in new_edges else 0 for p in nonedges], dtype=np.int8)

    base_scores = None
    rows = []
    for name, named_edges in periodic_edge_variants().items():
        S = periodic_matrix(named_edges)
        M = support_matrix(A, S)
        s_static = pair_scores(M, pairs)
        s_temp = pair_scores(M, nonedges)

        if base_scores is None:
            base_scores = s_static.copy()
        rho = float(spearmanr(base_scores, s_static).statistic)
        static_auc = float(roc_auc_score(y_static, s_static))
        temporal_auc = float(roc_auc_score(y_temp, s_temp))

        rows.append({
            'definition': name,
            'S_edges': len(named_edges),
            'spearman_rho_vs_baseline': rho,
            'static_auc': static_auc,
            'temporal_auc': temporal_auc,
        })

    result = pd.DataFrame(rows)
    result.to_csv(out / 'S_sensitivity_static_temporal_AUC.csv', index=False)
    print(result.to_string(index=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edges2025', default='2-1946.txt')
    ap.add_argument('--edges2026', default='2-2006.txt')
    ap.add_argument('--out', default='results/s_sensitivity')
    args = ap.parse_args()
    run(args.edges2025, args.edges2026, args.out)


if __name__ == '__main__':
    main()
