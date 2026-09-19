#!/usr/bin/env python3
"""Run all analyses reported in the manuscript with one command."""
from __future__ import annotations

import argparse
from pathlib import Path

import main_analysis
import permutation_test
import robustness_missing_edges
import s_sensitivity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edges2025', default='2-1946.txt')
    ap.add_argument('--edges2026', default='2-2006.txt')
    ap.add_argument('--out', default='results')
    ap.add_argument('--permutations', type=int, default=100000)
    ap.add_argument('--robustness-reps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260916)
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    print('\n### 1/4 Main analyses ###')
    main_analysis.run(args.edges2025, args.edges2026, str(root / 'main'))

    print('\n### 2/4 Permutation test ###')
    permutation_test.run(
        args.edges2025, args.edges2026,
        permutations=args.permutations, seed=args.seed,
        out_dir=str(root / 'permutation'),
    )

    print('\n### 3/4 Missing-edge robustness ###')
    robustness_missing_edges.run(
        args.edges2025, reps=args.robustness_reps, seed=args.seed,
        out_dir=str(root / 'robustness'),
    )

    print('\n### 4/4 S sensitivity ###')
    s_sensitivity.run(args.edges2025, args.edges2026, str(root / 's_sensitivity'))

    print(f'\nAll analyses completed. Results are in: {root.resolve()}')


if __name__ == '__main__':
    main()
