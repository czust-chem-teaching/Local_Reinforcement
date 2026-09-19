#!/usr/bin/env python3
"""Reproduce the main static and temporal analyses in Sections 3.1--3.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from lr_utils import (
    SYMBOLS, adjacency_from_edges, all_pairs, baseline_periodic_edges,
    load_edges, pair_scores, periodic_matrix, support_class, support_matrix,
    validate_temporal_edges,
)


def grouped_table(scores: np.ndarray, y: np.ndarray, positive_label: str) -> pd.DataFrame:
    cls = support_class(scores)
    rows = []
    labels = ['M=0', 'M=1', 'M=2', 'M>=3']
    for c, label in enumerate(labels):
        mask = cls == c
        total = int(mask.sum())
        positive = int(y[mask].sum())
        rows.append({
            'support_class': label,
            'total_pairs': total,
            positive_label: positive,
            'percent': 100.0 * positive / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def make_figure1(static_table: pd.DataFrame, out_pdf: Path) -> None:
    x = np.arange(4)
    y = static_table['percent'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, y, marker='o')
    ax.set_xticks(x)
    ax.set_xticklabels(['M=0', 'M=1', 'M=2', 'M≥3'])
    ax.set_xlabel('Local structural support')
    ax.set_ylabel('Prevalence of observed binary-compound relations (%)')
    ax.set_ylim(0, max(85.0, float(y.max()) + 6.0))
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1.5, f'{yi:.2f}%', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)


def run(edges2025: str, edges2026: str, out_dir: str = 'results') -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    e25 = load_edges(edges2025)
    e26 = load_edges(edges2026)
    new_edges = validate_temporal_edges(e25, e26)

    A = adjacency_from_edges(e25)
    S = periodic_matrix(baseline_periodic_edges())
    M = support_matrix(A, S)

    pairs = all_pairs()
    nonedges = [p for p in pairs if p not in e25]

    # Section 3.1: static prevalence over all 5253 pairs.
    y_static = np.asarray([1 if p in e25 else 0 for p in pairs], dtype=np.int8)
    s_static = pair_scores(M, pairs)
    static_tbl = grouped_table(s_static, y_static, 'observed_2025_edges')
    static_tbl.to_csv(out / 'static_support_prevalence.csv', index=False)
    make_figure1(static_tbl, out / 'Figure1_local_support_prevalence.pdf')

    # Section 3.2: temporal appearance among 2025 nonedges.
    y_temp = np.asarray([1 if p in new_edges else 0 for p in nonedges], dtype=np.int8)
    s_temp = pair_scores(M, nonedges)
    temporal_tbl = grouped_table(s_temp, y_temp, 'later_recorded_edges')
    temporal_tbl.to_csv(out / 'temporal_support_appearance.csv', index=False)

    # Logistic effect size, using raw M. Parametric p-value is not used as the main significance test.
    X = sm.add_constant(s_temp)
    fit = sm.Logit(y_temp, X).fit(disp=False)
    beta = float(fit.params[1])
    odds_ratio = float(np.exp(beta))

    # Section 3.3: AUC comparison.
    deg = A.sum(axis=1).astype(float)
    A2 = A @ A
    pa = np.asarray([deg[i] * deg[j] for i, j in nonedges], dtype=float)
    cn = np.asarray([A2[i, j] for i, j in nonedges], dtype=float)
    auc_local = float(roc_auc_score(y_temp, s_temp))
    auc_pa = float(roc_auc_score(y_temp, pa))
    auc_cn = float(roc_auc_score(y_temp, cn))
    auc_tbl = pd.DataFrame([
        {'score': 'Local support M=(SA+AS)', 'AUC': auc_local},
        {'score': 'Preferential attachment k_i*k_j', 'AUC': auc_pa},
        {'score': 'Common neighbors (A^2)_ij', 'AUC': auc_cn},
    ])
    auc_tbl.to_csv(out / 'auc_comparison.csv', index=False)

    summary = {
        'n_elements': len(SYMBOLS),
        'n_possible_pairs': len(pairs),
        'edges_2025': len(e25),
        'edges_2026': len(e26),
        'later_recorded_edges': len(new_edges),
        'nonedges_2025': len(nonedges),
        'periodic_table_edges_baseline': int(S.sum() // 2),
        'logistic_beta_per_M': beta,
        'odds_ratio_per_M': odds_ratio,
        'temporal_auc_local_support': auc_local,
        'temporal_auc_preferential_attachment': auc_pa,
        'temporal_auc_common_neighbors': auc_cn,
        'static_auc_local_support': float(roc_auc_score(y_static, s_static)),
    }
    (out / 'main_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('=== Data checks ===')
    print(f'Elements: {len(SYMBOLS)}')
    print(f'Possible pairs: {len(pairs)}')
    print(f'2025 edges: {len(e25)}')
    print(f'2026 edges: {len(e26)}')
    print(f'Later-recorded edges: {len(new_edges)}')
    print(f'2025 nonedges: {len(nonedges)}')
    print(f'Baseline S edges: {int(S.sum() // 2)}')
    print('\n=== Static prevalence ===')
    print(static_tbl.to_string(index=False))
    print('\n=== Temporal appearance ===')
    print(temporal_tbl.to_string(index=False))
    print('\n=== Logistic effect size ===')
    print(f'beta = {beta:.6f}; odds ratio = {odds_ratio:.6f}')
    print('\n=== Temporal AUC comparison ===')
    print(auc_tbl.to_string(index=False))
    print(f'\nOutputs written to: {out.resolve()}')
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edges2025', default='2-1946.txt', help='2025 edge-list file (default: 2-1946.txt)')
    ap.add_argument('--edges2026', default='2-2006.txt', help='2026 edge-list file (default: 2-2006.txt)')
    ap.add_argument('--out', default='results/main', help='Output directory')
    args = ap.parse_args()
    run(args.edges2025, args.edges2026, args.out)


if __name__ == '__main__':
    main()
