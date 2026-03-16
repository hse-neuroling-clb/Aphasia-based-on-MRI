import argparse
import os

import numpy as np
import pandas as pd

from analysis_common import (
    HYPERPARAMETER_COMPLEXITY,
    METRIC_WEIGHTS,
    RGI_FILTERING_CONFIG,
    RGI_METRICS,
    default_results_dir,
    infer_optimization_metric,
    infer_smote_flag,
    load_config_data,
    require_manifest,
    validate_complete_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute RGI rankings from saved model result tables.")
    parser.add_argument(
        "--results-dir",
        default=default_results_dir(),
        help="Directory containing one subfolder per configuration.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for RGI CSV outputs. Defaults to results-dir.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to experiment_manifest.json. Defaults to results-dir/experiment_manifest.json.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional subset of configuration folder names.",
    )
    return parser.parse_args()


def calculate_rgi(avg_df, std_df, hyperparameter_dict, filtering_config):
    models = avg_df.index.tolist()
    initial_count = len(models)

    blacklist = filtering_config["baseline_blacklist"]
    models = [model for model in models if model not in blacklist]

    screening_metrics = filtering_config["screening_metrics"]
    available_screening = [metric for metric in screening_metrics if metric in avg_df.columns]
    if available_screening:
        screening_df = avg_df.loc[models, available_screening]
        avg_pct = screening_df.rank(pct=True).mean(axis=1)
        qualified = avg_pct[avg_pct >= filtering_config["percentile_threshold"]].index.tolist()
        if len(qualified) >= filtering_config["min_models"]:
            models = qualified

    avg_filtered = avg_df.loc[models]
    std_filtered = std_df.loc[models]
    n_models = len(models)
    available_rgi_metrics = [metric for metric in RGI_METRICS if metric in avg_filtered.columns]

    composite_perf = pd.Series(0.0, index=models)
    composite_std = pd.Series(0.0, index=models)

    for metric in available_rgi_metrics:
        weight = METRIC_WEIGHTS.get(metric, 1.0)

        perf_rank = avg_filtered[metric].rank(ascending=False, method="average")
        composite_perf += weight * (n_models + 1 - perf_rank)

        if metric in std_filtered.columns:
            std_rank = std_filtered[metric].rank(ascending=True, method="average")
            composite_std += weight * (n_models + 1 - std_rank)

    min_perf = composite_perf.min()
    max_perf = composite_perf.max()
    if max_perf > min_perf:
        perf_norm = (composite_perf - min_perf) / (max_perf - min_perf)
    else:
        perf_norm = pd.Series(1.0, index=models)

    min_std = composite_std.min()
    max_std = composite_std.max()
    if max_std > min_std:
        var_penalty = (composite_std - min_std) / (max_std - min_std)
    else:
        var_penalty = pd.Series(1.0, index=models)

    hp_counts = pd.Series([hyperparameter_dict.get(model, 5) for model in models], index=models)
    log_hp = np.log1p(hp_counts)
    min_log = log_hp.min()
    max_log = log_hp.max()
    if max_log > min_log:
        normalized_hp = (log_hp - min_log) / (max_log - min_log)
        hps_penalty = 1.0 - 0.7 * normalized_hp
    else:
        hps_penalty = pd.Series(1.0, index=models)

    var_std = var_penalty.std()
    hps_std = hps_penalty.std()
    total_std = var_std + hps_std
    if total_std > 1e-10:
        w_var = var_std / total_std
        w_hps = hps_std / total_std
    else:
        w_var = 0.5
        w_hps = 0.5

    stability_factor = w_var * var_penalty + w_hps * hps_penalty
    rgi_score = (perf_norm ** 0.6) * (stability_factor ** 0.4)

    result = pd.DataFrame(
        {
            "model": models,
            "rgi_score": rgi_score.values,
            "perf_norm": perf_norm.values,
            "stability_factor": stability_factor.values,
            "var_penalty": var_penalty.values,
            "hps_penalty": hps_penalty.values,
            "composite_performance": composite_perf.values,
            "composite_std": composite_std.values,
            "hyperparameter_count": hp_counts.values,
        }
    )

    for metric in available_rgi_metrics:
        result[metric] = avg_filtered.loc[models, metric].values

    result = result.sort_values("rgi_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    print(f"{initial_count} -> {len(models)} models kept for RGI")
    return result


def aggregate_robustness(combined_rgi):
    model_robustness = (
        combined_rgi.groupby("model")
        .agg(
            {
                "rgi_score": ["mean", "std", "min", "max"],
                "rank": ["mean", "std"],
                "composite_performance": "mean",
                "stability_factor": "mean",
                "hyperparameter_count": "first",
                "config_folder": "count",
            }
        )
        .reset_index()
    )

    model_robustness.columns = [
        "model",
        "rgi_mean",
        "rgi_std",
        "rgi_min",
        "rgi_max",
        "rank_mean",
        "rank_std",
        "avg_composite_performance",
        "avg_stability_factor",
        "hyperparameter_count",
        "n_configs",
    ]

    top3_counts = (
        combined_rgi[combined_rgi["rank"] <= 3]
        .groupby("model")
        .size()
        .reset_index(name="top3_appearances")
    )
    top1_counts = (
        combined_rgi[combined_rgi["rank"] == 1]
        .groupby("model")
        .size()
        .reset_index(name="top1_appearances")
    )

    model_robustness = model_robustness.merge(top3_counts, on="model", how="left")
    model_robustness = model_robustness.merge(top1_counts, on="model", how="left")
    model_robustness["top3_appearances"] = model_robustness["top3_appearances"].fillna(0).astype(int)
    model_robustness["top1_appearances"] = model_robustness["top1_appearances"].fillna(0).astype(int)
    model_robustness["rgi_std"] = model_robustness["rgi_std"].fillna(0.0)
    model_robustness["rank_std"] = model_robustness["rank_std"].fillna(0.0)

    model_robustness = model_robustness.sort_values("rgi_mean", ascending=False).reset_index(drop=True)
    model_robustness["overall_rank"] = range(1, len(model_robustness) + 1)
    return model_robustness


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir or results_dir)
    manifest = require_manifest(results_dir, args.manifest)
    validate_complete_results(manifest)
    config_folders = args.configs or [cfg["config_name"] for cfg in manifest["expected_configs"]]

    os.makedirs(out_dir, exist_ok=True)

    all_rgi = []
    for config_folder in config_folders:
        avg_df, std_df = load_config_data(results_dir, config_folder)
        if avg_df.empty:
            continue

        print(f"Processing {config_folder}")
        config_rgi = calculate_rgi(avg_df, std_df, HYPERPARAMETER_COMPLEXITY, RGI_FILTERING_CONFIG)
        config_rgi["config_folder"] = config_folder
        config_rgi["smote"] = infer_smote_flag(config_folder)
        config_rgi["optimization_metric"] = infer_optimization_metric(config_folder)
        all_rgi.append(config_rgi)

    if not all_rgi:
        raise ValueError("No RGI tables were produced.")

    combined_rgi = pd.concat(all_rgi, ignore_index=True)
    model_robustness = aggregate_robustness(combined_rgi)

    by_config_path = os.path.join(out_dir, "final_rgi_results_by_config.csv")
    robustness_path = os.path.join(out_dir, "final_rgi_model_robustness.csv")
    combined_rgi.to_csv(by_config_path, index=False)
    model_robustness.to_csv(robustness_path, index=False)

    print(f"Saved {by_config_path}")
    print(f"Saved {robustness_path}")


if __name__ == "__main__":
    main()
