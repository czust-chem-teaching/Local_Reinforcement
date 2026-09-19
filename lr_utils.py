#!/usr/bin/env python3
"""Shared utilities for the local-reinforcement analyses.

Input edge files are plain text with one unordered element pair per line, e.g.
    Br,Pt
    F,Ga
Whitespace-separated pairs are also accepted. Duplicate/self-loop/unknown-element
records raise an error so that the analysis fails loudly rather than silently changing data.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

SYMBOLS: List[str] = [
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
    'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'
]
N = len(SYMBOLS)
INDEX: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
Pair = Tuple[int, int]
NamedPair = Tuple[str, str]


def canonical_pair(i: int, j: int) -> Pair:
    if i == j:
        raise ValueError(f"Self-loop is not allowed: {SYMBOLS[i]}")
    return (i, j) if i < j else (j, i)


def load_edges(path: str | Path) -> Set[Pair]:
    """Load an unordered edge list of element symbols.

    Accepted line formats include ``Br,Pt`` and ``Br Pt``. Blank lines and lines
    beginning with ``#`` are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    edges: Set[Pair] = set()
    n_records = 0
    for line_no, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = [x for x in re.split(r"[,;\s]+", line) if x]
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected two element symbols, got {raw!r}")
        a, b = parts
        if a not in INDEX or b not in INDEX:
            raise ValueError(f"{path}:{line_no}: unknown element symbol in {raw!r}")
        edge = canonical_pair(INDEX[a], INDEX[b])
        n_records += 1
        if edge in edges:
            raise ValueError(f"{path}:{line_no}: duplicate edge {a},{b}")
        edges.add(edge)

    if len(edges) != n_records:
        raise AssertionError("Unexpected duplicate handling error")
    return edges


def adjacency_from_edges(edges: Iterable[Pair], dtype=np.int16) -> np.ndarray:
    A = np.zeros((N, N), dtype=dtype)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    return A


def all_pairs() -> List[Pair]:
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def baseline_periodic_edges() -> Set[NamedPair]:
    """Baseline 148-edge periodic-table adjacency used in the manuscript."""
    main: Dict[str, Tuple[int, int]] = {'H': (1, 1), 'He': (1, 18)}
    for s, g in zip(['Li','Be','B','C','N','O','F','Ne'], [1,2,13,14,15,16,17,18]):
        main[s] = (2, g)
    for s, g in zip(['Na','Mg','Al','Si','P','S','Cl','Ar'], [1,2,13,14,15,16,17,18]):
        main[s] = (3, g)
    for s, g in zip(['K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr'], range(1, 19)):
        main[s] = (4, g)
    for s, g in zip(['Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe'], range(1, 19)):
        main[s] = (5, g)
    for s, g in zip(['Cs','Ba','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn'], [1,2] + list(range(4, 19))):
        main[s] = (6, g)
    for s, g in zip(['Fr','Ra'], [1, 2]):
        main[s] = (7, g)

    E: Set[NamedPair] = set()

    # Horizontal adjacency in the main 18-group table.
    by_period: Dict[int, List[Tuple[int, str]]] = {}
    for s, (p, g) in main.items():
        by_period.setdefault(p, []).append((g, s))
    for arr in by_period.values():
        arr = sorted(arr)
        for (g1, s1), (g2, s2) in zip(arr, arr[1:]):
            if g2 - g1 == 1:
                E.add(tuple(sorted((s1, s2))))

    # Short-period bridges used by the baseline representation.
    for e in [('H','He'), ('Be','B'), ('Mg','Al')]:
        E.add(tuple(sorted(e)))

    # Vertical adjacency in the main table.
    by_group: Dict[int, List[Tuple[int, str]]] = {}
    for s, (p, g) in main.items():
        by_group.setdefault(g, []).append((p, s))
    for arr in by_group.values():
        arr = sorted(arr)
        for (p1, s1), (p2, s2) in zip(arr, arr[1:]):
            if p2 - p1 == 1:
                E.add(tuple(sorted((s1, s2))))

    # Detached f-block: horizontal consecutive elements only.
    lan = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
    act = ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
    for row in (lan, act):
        for a, b in zip(row, row[1:]):
            E.add(tuple(sorted((a, b))))

    if len(E) != 148:
        raise AssertionError(f"Baseline S should have 148 edges, got {len(E)}")
    return E


def periodic_edge_variants() -> Dict[str, Set[NamedPair]]:
    """Return the four S specifications used in the sensitivity analysis."""
    base = baseline_periodic_edges()
    special = {tuple(sorted(e)) for e in [('H','He'), ('Be','B'), ('Mg','Al')]}
    lan = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
    act = ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
    f_horizontal = {tuple(sorted(e)) for row in (lan, act) for e in zip(row, row[1:])}
    aligned = {tuple(sorted((a, b))) for a, b in zip(lan, act)}
    expanded_extra = aligned | {tuple(sorted(e)) for e in [('Ba','La'), ('Lu','Hf'), ('Ra','Ac')]}

    variants = {
        'Baseline': base,
        'No special bridges': base - special,
        'No f-block horizontal links': base - f_horizontal,
        'Expanded f-block layout': base | expanded_extra,
    }
    expected = {
        'Baseline': 148,
        'No special bridges': 145,
        'No f-block horizontal links': 120,
        'Expanded f-block layout': 166,
    }
    for name, E in variants.items():
        if len(E) != expected[name]:
            raise AssertionError(f"{name}: expected {expected[name]} S edges, got {len(E)}")
    return variants


def periodic_matrix(named_edges: Iterable[NamedPair]) -> np.ndarray:
    S = np.zeros((N, N), dtype=np.int16)
    for a, b in named_edges:
        i, j = INDEX[a], INDEX[b]
        S[i, j] = S[j, i] = 1
    return S


def support_matrix(A: np.ndarray, S: np.ndarray) -> np.ndarray:
    return S @ A + A @ S


def pair_scores(M: np.ndarray, pairs: Sequence[Pair]) -> np.ndarray:
    return np.asarray([M[i, j] for i, j in pairs], dtype=float)


def support_class(scores: np.ndarray) -> np.ndarray:
    """Map raw M values to 0,1,2,3 where 3 means M>=3."""
    return np.minimum(scores.astype(int), 3)


def validate_temporal_edges(e25: Set[Pair], e26: Set[Pair]) -> Set[Pair]:
    missing = e25 - e26
    if missing:
        ex = sorted(missing)[:5]
        text = ', '.join(f"{SYMBOLS[i]}-{SYMBOLS[j]}" for i, j in ex)
        raise ValueError(f"2025 edge set is not contained in 2026; examples: {text}")
    return e26 - e25
