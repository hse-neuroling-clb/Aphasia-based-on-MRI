import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from analysis_common import (
    build_config_name,
    format_mean_std,
    parse_bool,
    result_csv_path,
    save_manifest,
    selection_to_model_names,
    split_avg_std,
    validate_single_row_csv,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SEARCH_DIR = os.path.join(BASE_DIR, "model_search")
DEFAULT_CONFIG_PATH = os.path.join(MODEL_SEARCH_DIR, "config.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the portable experiment matrix and aggregate repeated runs.")
    parser.add_argument(
        "--workspace",
        default=os.path.join(BASE_DIR, "portable_workspace"),
        help="Workspace root for raw runs and aggregated results. Default: %(default)s",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Base YAML config used by model_search/main.py. Default: %(default)s",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Input CSV path. If omitted, the path in config.yaml is used.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "balanced_accuracy", "f1_weighted", "mcc"],
        help="Optimization metrics to run. Default: %(default)s",
    )
    parser.add_argument(
        "--oversampling",
        nargs="+",
        default=["true", "false"],
        help="SMOTE settings to run, using true/false values. Default: %(default)s",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of repeated runs per metric/SMOTE configuration. Default: %(default)s",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Seed used for run 1. Default: %(default)s",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment applied to the seed between repeated runs. Default: %(default)s",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optuna trial count override. If omitted, the value in config.yaml is used.",
    )
    parser.add_argument(
        "--tabpfn-device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="TabPFN device override. If omitted, the value in config.yaml is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the effective plan and exit without running experiments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed raw runs instead of skipping them.",
    )
    return parser.parse_args()


def load_base_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_manifest(args, base_config, config_path, data_path, workspace, results_dir, raw_runs_dir):
    selected_models = base_config["model_selection"]["selected_models"]
    expected_models = selection_to_model_names(
        selected_models,
        all_models=base_config["model_selection"]["all_models"],
    )

    expected_configs = []
    for metric in args.metrics:
        for oversampling in [parse_bool(value) for value in args.oversampling]:
            config_name = build_config_name(metric, oversampling, args.n_trials or base_config["number_of_trials"], args.n_runs)
            run_dirs = []
            run_seeds = []
            for run_idx in range(1, args.n_runs + 1):
                run_seed = args.seed_base + (run_idx - 1) * args.seed_step
                run_root = os.path.join(raw_runs_dir, config_name, f"run_{run_idx:02d}")
                run_dirs.append(os.path.join(run_root, config_name))
                run_seeds.append(run_seed)

            expected_configs.append(
                {
                    "config_name": config_name,
                    "metric": metric,
                    "oversampling": oversampling,
                    "summary_dir": os.path.join(results_dir, config_name),
                    "run_dirs": run_dirs,
                    "run_seeds": run_seeds,
                }
            )

    return {
        "manifest_path": os.path.join(results_dir, "experiment_manifest.json"),
        "workspace": workspace,
        "results_dir": results_dir,
        "raw_runs_dir": raw_runs_dir,
        "config_path": config_path,
        "data_path": data_path,
        "selected_model_keys": selected_models,
        "n_runs": args.n_runs,
        "seed_base": args.seed_base,
        "seed_step": args.seed_step,
        "n_trials": args.n_trials or base_config["number_of_trials"],
        "target_combination": base_config["combinations"][base_config["start"]],
        "cross_validation": base_config["cross_validation"],
        "tabpfn_device": args.tabpfn_device or base_config.get("tabpfn_device", "auto"),
        "expected_models": expected_models,
        "expected_configs": expected_configs,
    }


def print_execution_plan(manifest):
    metric_order = []
    for cfg in manifest["expected_configs"]:
        if cfg["metric"] not in metric_order:
            metric_order.append(cfg["metric"])

    print("=" * 80)
    print("Portable Experiment Plan")
    print("=" * 80)
    print(f"workspace: {manifest['workspace']}")
    print(f"results_dir: {manifest['results_dir']}")
    print(f"data_path: {manifest['data_path']}")
    print(f"config_path: {manifest['config_path']}")
    print(f"metrics: {metric_order}")
    print(f"smote: {sorted(list({cfg['oversampling'] for cfg in manifest['expected_configs']}))}")
    print(f"n_runs: {manifest['n_runs']}")
    print(f"seed_base: {manifest['seed_base']}")
    print(f"seed_step: {manifest['seed_step']}")
    print(f"n_trials: {manifest['n_trials']}")
    print(f"tabpfn_device: {manifest['tabpfn_device']}")
    print(f"cv: {manifest['cross_validation']}")
    print(f"target_combination: {manifest['target_combination']}")
    print(f"selected_models: {manifest['selected_model_keys']}")
    print(f"output_model_names: {manifest['expected_models']}")
    print(f"n_configurations: {len(manifest['expected_configs'])}")
    print("-" * 80)
    for cfg in manifest["expected_configs"]:
        seeds = cfg["run_seeds"]
        print(
            f"{cfg['config_name']}: metric={cfg['metric']}, smote={cfg['oversampling']}, "
            f"runs={len(seeds)}, seeds={seeds[0]}..{seeds[-1]}"
        )
    print("=" * 80)


def is_complete_run(run_dir, config_name, expected_models):
    if not os.path.isdir(run_dir):
        return False
    for model_name in expected_models:
        csv_path = result_csv_path(run_dir, config_name, model_name)
        if not os.path.isfile(csv_path):
            return False
        try:
            validate_single_row_csv(csv_path)
        except ValueError:
            return False
    return True


def choose_best_run(rows, metric_name):
    scores = []
    for row in rows:
        score, _ = split_avg_std(row[metric_name])
        scores.append(score)
    return int(max(range(len(scores)), key=lambda idx: scores[idx]))


def aggregate_config(config_meta, expected_models):
    config_name = config_meta["config_name"]
    summary_dir = config_meta["summary_dir"]
    os.makedirs(summary_dir, exist_ok=True)

    for model_name in expected_models:
        rows = []
        for run_dir in config_meta["run_dirs"]:
            csv_path = result_csv_path(run_dir, config_name, model_name)
            df = validate_single_row_csv(csv_path)
            rows.append(df.iloc[0])

        best_run_idx = choose_best_run(rows, config_meta["metric"])
        summary_row = {}

        for column in rows[0].index:
            values = [row[column] for row in rows]

            if column == "idx":
                if len(set(values)) != 1:
                    raise ValueError(f"Inconsistent idx values for {config_name}/{model_name}")
                summary_row[column] = values[0]
                continue

            if column == "combination":
                if len(set(str(value) for value in values)) != 1:
                    raise ValueError(f"Inconsistent combinations for {config_name}/{model_name}")
                summary_row[column] = values[0]
                continue

            if column == "best_params":
                summary_row[column] = values[best_run_idx]
                continue

            means = []
            for value in values:
                avg, _ = split_avg_std(value)
                if np.isnan(avg):
                    raise ValueError(f"Cannot aggregate column {column} in {config_name}/{model_name}")
                means.append(avg)
            summary_row[column] = format_mean_std(means)

        output_path = result_csv_path(summary_dir, config_name, model_name)
        pd.DataFrame([summary_row]).to_csv(output_path)


def run_single_config(config_meta, manifest, args):
    for run_idx, (run_dir, run_seed) in enumerate(zip(config_meta["run_dirs"], config_meta["run_seeds"]), start=1):
        if not args.force and is_complete_run(run_dir, config_meta["config_name"], manifest["expected_models"]):
            print(f"Skipping completed run: {config_meta['config_name']} / run_{run_idx:02d}")
            continue

        run_root = os.path.dirname(run_dir)
        os.makedirs(run_root, exist_ok=True)

        command = [
            sys.executable,
            os.path.join(MODEL_SEARCH_DIR, "main.py"),
            "--config",
            manifest["config_path"],
            "--data-path",
            manifest["data_path"],
            "--scoring",
            config_meta["metric"],
            "--oversampling",
            str(config_meta["oversampling"]).lower(),
            "--seed",
            str(run_seed),
            "--n-trials",
            str(manifest["n_trials"]),
            "--output-root",
            run_root,
            "--exp-name",
            config_meta["config_name"],
        ]

        if args.tabpfn_device is not None:
            command.extend(["--tabpfn-device", args.tabpfn_device])

        print(f"Running {config_meta['config_name']} / run_{run_idx:02d}")
        subprocess.run(command, check=True, cwd=MODEL_SEARCH_DIR)

        if not is_complete_run(run_dir, config_meta["config_name"], manifest["expected_models"]):
            raise RuntimeError(f"Incomplete run output detected: {run_dir}")

    aggregate_config(config_meta, manifest["expected_models"])


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    data_path = os.path.abspath(args.data_path) if args.data_path else None
    workspace = os.path.abspath(args.workspace)
    results_dir = os.path.join(workspace, "results")
    raw_runs_dir = os.path.join(workspace, "raw_runs")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(raw_runs_dir, exist_ok=True)

    base_config = load_base_config(config_path)
    if base_config["end"] - base_config["start"] != 1:
        raise ValueError("Portable run_experiment.py requires exactly one target combination in the config file.")

    if data_path is None:
        cfg_data_path = base_config["data_path"]
        if os.path.isabs(cfg_data_path):
            data_path = cfg_data_path
        else:
            data_path = os.path.abspath(os.path.join(os.path.dirname(config_path), cfg_data_path))

    manifest = build_manifest(args, base_config, config_path, data_path, workspace, results_dir, raw_runs_dir)
    save_manifest(manifest)
    print_execution_plan(manifest)

    if args.dry_run:
        print("Dry run only. No experiments were executed.")
        return

    for config_meta in manifest["expected_configs"]:
        run_single_config(config_meta, manifest, args)
        save_manifest(manifest)

    print(f"Workspace ready: {workspace}")
    print(f"Manifest saved: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
