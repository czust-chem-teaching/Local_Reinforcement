# Local Reinforcement in Binary-Compound Networks

Reproducibility code for the manuscript:

**Local reinforcement in the growth of periodic-table-structured binary compound networks**

## Repository layout

All data files and Python scripts are kept in the **same directory**. No `data/` or `code/` subfolders are required.

```text
Local_Reinforcement/
├── 2-1946.txt
├── 2-2006.txt
├── README.md
├── requirements.txt
├── lr_utils.py
├── main_analysis.py
├── permutation_test.py
├── robustness_missing_edges.py
├── s_sensitivity.py
└── run_all.py
```

The two data files are:

- `2-1946.txt` — ChemSpider snapshot collected on April 4, 2025, containing 1,946 recorded binary elemental relations.
- `2-2006.txt` — ChemSpider snapshot collected on June 18, 2026, containing 2,006 recorded binary elemental relations.

Each non-empty line contains one unordered elemental pair. Both comma-separated and whitespace-separated formats are accepted, for example:

```text
Br,Pt
F,Ga
```

The scripts check element symbols, self-loops, duplicate edges, and whether the 2025 edge set is contained in the 2026 edge set.

## Requirements

Python 3.10 or newer is recommended.

Install the required packages from the repository directory:

```bash
pip install -r requirements.txt
```

Required packages are listed in `requirements.txt` and include NumPy, pandas, SciPy, scikit-learn, statsmodels, and Matplotlib.

## Reproduce all analyses

Because the data files and scripts are in the same directory, simply run:

```bash
python run_all.py
```

The default input files are `2-1946.txt` and `2-2006.txt`.

The script creates a `results/` directory and runs the four analyses reported in the manuscript:

1. **Main analysis** — static support prevalence, 2025-to-2026 temporal appearance, logistic odds ratio, comparison with preferential attachment and common neighbors, and Figure 1 in PDF format.
2. **Permutation test** — 100,000 random-label permutations of the 60 later-recorded relations among the 3,307 candidate nonedges.
3. **Missing-edge robustness** — random removal of 5%, 10%, 20%, and 30% of the 2025 recorded relations, with 1,000 realizations at each removal level.
4. **Periodic-table adjacency sensitivity** — baseline and three alternative definitions of the periodic-table adjacency network `S`, reporting Spearman correlation, static AUC, and temporal AUC.

The default random seed is `20260916`.

If desired, alternative file names can still be supplied explicitly:

```bash
python run_all.py --edges2025 another_2025_file.txt --edges2026 another_2026_file.txt --out results
```

## Run individual analyses

### Main analyses: Sections 3.1--3.3

```bash
python main_analysis.py
```

Main outputs are written to `results/main/`, including:

- `static_support_prevalence.csv`
- `temporal_support_appearance.csv`
- `auc_comparison.csv`
- `main_summary.json`
- `Figure1_local_support_prevalence.pdf`

### Permutation test

```bash
python permutation_test.py
```

Defaults: 100,000 permutations and random seed `20260916`.

The test keeps the 2025 local-support values fixed and randomly reassigns the 60 later-recorded positive labels among all 3,307 pairs that are nonedges in 2025. It is therefore a **random-label null model**. It is not a degree-preserving or node-dependence-preserving permutation.

Outputs are written to `results/permutation/`.

### Robustness to incomplete compound records

```bash
python robustness_missing_edges.py
```

The script randomly removes 5%, 10%, 20%, and 30% of the 1,946 relations in the 2025 network and repeats each removal experiment 1,000 times. Removed relations are treated as hidden positives, while original 2025 nonedges are treated as negatives.

Outputs are written to `results/robustness/`.

### Sensitivity to periodic-table adjacency

```bash
python s_sensitivity.py
```

Four definitions of `S` are evaluated:

- baseline periodic-table adjacency: 148 edges;
- baseline without H--He, Be--B, and Mg--Al: 145 edges;
- baseline without horizontal lanthanide and actinide links: 120 edges;
- expanded f-block layout: 166 edges.

Outputs are written to `results/s_sensitivity/`.

## Expected manuscript-level results

Using `2-1946.txt` and `2-2006.txt`, the scripts should reproduce the principal results reported in the manuscript, up to printed rounding:

- 103 elements and 5,253 possible unordered elemental pairs;
- 1,946 relations in the 2025 snapshot and 2,006 in the 2026 snapshot;
- 60 later-recorded relations;
- baseline periodic-table adjacency network `S`: 148 edges;
- static prevalence for `M=0`, `M=1`, `M=2`, and `M>=3`: approximately 1.96%, 8.40%, 30.69%, and 77.75%;
- temporal appearance probabilities: approximately 0.31%, 1.33%, 2.74%, and 6.95%;
- logistic odds ratio for a one-unit increase in local support: approximately 2.67;
- temporal AUC: local support 0.801, preferential attachment 0.725, common neighbors 0.648;
- permutation-test result: observed AUC approximately 0.801 and `p_perm < 1e-5` for 100,000 permutations;
- temporal AUC under the four `S` definitions: approximately 0.801, 0.798, 0.751, and 0.846.

## Interpretation of the temporal data

The 60 relations present in the 2026 snapshot but absent in the 2025 snapshot are treated as **later-recorded relations**. Their presence in the later database snapshot does not establish that the corresponding compounds were first discovered between the two collection dates.

## Citation

If you use these data or scripts, please cite the associated manuscript once its bibliographic information is available.
