# Code Availability

This repository contains the analysis scripts accompanying the manuscript submission.

The package includes:

- `model_search/`: model fitting and hyperparameter search
- `run_experiment.py`: repeated experiment execution and result aggregation
- `run_votenrank.py`: VoteNRank-based model ranking
- `run_rgi.py`: Robust Generalization Index analysis

## Minimal Workflow

1. Run the experiment workspace:

```bash
python run_experiment.py --data-path /path/to/Chronic_Data_Clean.csv
```

2. Run VoteNRank analysis:

```bash
python run_votenrank.py --results-dir portable_workspace/results
```

3. Run RGI analysis:

```bash
python run_rgi.py --results-dir portable_workspace/results
```

## Configuration

Base experiment settings are defined in `model_search/config.yaml`.

Frequently adjusted settings can also be overridden from the command line in `run_experiment.py`, including:

- input data path
- number of repeated runs
- optimization metric set
- SMOTE setting
- trial count

To inspect the effective run plan without starting the experiment:

```bash
python run_experiment.py --data-path /path/to/Chronic_Data_Clean.csv --dry-run
```

For the full list of command-line options:

```bash
python run_experiment.py --help
```

## Outputs

`run_experiment.py` creates a portable workspace containing:

- raw run outputs in `portable_workspace/raw_runs/`
- aggregated summary tables in `portable_workspace/results/`

`run_votenrank.py` writes:

- `votenrank_winners.csv`
- `votenrank_full_ranking.csv`

`run_rgi.py` writes:

- `final_rgi_results_by_config.csv`
- `final_rgi_model_robustness.csv`

## Note

`run_votenrank.py` and `run_rgi.py` require a complete result workspace and will stop if expected result files are missing.

The data file is not bundled in this repository.
