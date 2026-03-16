import glob
import json
import os
import re

import numpy as np
import pandas as pd


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "mcc",
    "cohen_kappa",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
]

RGI_METRICS = [
    "balanced_accuracy",
    "f1_weighted",
    "mcc",
    "cohen_kappa",
    "roc_auc_ovo_weighted",
]

MODEL_OUTPUT_NAMES = {
    "RandomLabeling": "RandomLabeling",
    "MajorityVote": "MajorityVote",
    "DecisionStump": "DecisionStump",
    "NaiveBayes": "GaussianNB",
    "QDA": "QuadraticDiscriminantAnalysis",
    "KNN": "KNeighborsClassifier",
    "RKNN": "RadiusNeighborsClassifier",
    "LR": "LogisticRegression",
    "RR": "RidgeClassifierWithProba",
    "SVC_linear": "SVC_linear",
    "SVC_poly": "SVC_poly",
    "SVC_sigmoid": "SVC_sigmoid",
    "SVC_rbf": "SVC_rbf",
    "DT": "DecisionTreeClassifier",
    "Adaboost": "AdaBoostClassifier",
    "RF": "RandomForestClassifier",
    "MLP": "MLPClassifier",
    "XGB": "XGBClassifier",
    "LGBM": "LGBMClassifier",
    "Tab": "TabPFNClassifier",
}

HYPERPARAMETER_COMPLEXITY = {
    "XGBClassifier": 10,
    "LGBMClassifier": 10,
    "RandomForestClassifier": 8,
    "TabPFNClassifier": 2,
    "MLPClassifier": 7,
    "SVC_rbf": 6,
    "SVC_poly": 7,
    "SVC_sigmoid": 6,
    "SVC_linear": 4,
    "LogisticRegression": 3,
    "AdaBoostClassifier": 5,
    "KNeighborsClassifier": 4,
    "DecisionTreeClassifier": 6,
    "GaussianNB": 2,
    "QuadraticDiscriminantAnalysis": 2,
    "RidgeClassifierWithProba": 3,
    "DecisionStump": 2,
    "RadiusNeighborsClassifier": 5,
    "MajorityVote": 1,
    "RandomLabeling": 1,
}

RGI_FILTERING_CONFIG = {
    "baseline_blacklist": ["MajorityVote", "RandomLabeling"],
    "percentile_threshold": 0.30,
    "screening_metrics": ["balanced_accuracy", "f1_weighted", "mcc"],
    "min_models": 5,
}

METRIC_WEIGHTS = {
    "balanced_accuracy": 1.0,
    "f1_weighted": 1.0,
    "mcc": 1.1,
    "cohen_kappa": 1.0,
    "roc_auc_ovo_weighted": 1.0,
}


def default_results_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "portable_workspace", "results"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "..", "Model_Search", "results"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


def split_avg_std(value):
    if isinstance(value, str) and "±" in value:
        avg, std = value.split("±", 1)
        return float(avg.strip()), float(std.strip())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), np.nan
    return np.nan, np.nan


def format_mean_std(values):
    array = np.asarray(values, dtype=float)
    return f"{np.mean(array):.3f} ± {np.std(array, ddof=0):.3f}"


def extract_model_name(filename, config_folder):
    prefix = config_folder + "_"
    basename = os.path.splitext(filename)[0]
    if basename.startswith(prefix):
        return basename[len(prefix):]
    return None


def result_csv_name(config_name, model_name):
    return f"{config_name}_{model_name}.csv"


def result_csv_path(folder_path, config_name, model_name):
    return os.path.join(folder_path, result_csv_name(config_name, model_name))


def selection_to_model_names(selected_models, all_models=False):
    if all_models:
        return list(MODEL_OUTPUT_NAMES.values())
    return [MODEL_OUTPUT_NAMES[name] for name in selected_models]


def build_config_name(metric, oversampling, n_trials, n_runs):
    return f"Severity_Max_{metric}_SMOTE={oversampling}_Trails={n_trials}_Runs={n_runs}"


def discover_config_folders(results_dir):
    config_folders = []
    if not os.path.isdir(results_dir):
        return config_folders

    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        if entry.startswith(".") or entry == "best_models":
            continue
        if glob.glob(os.path.join(path, "*.csv")):
            config_folders.append(entry)
    return config_folders


def load_config_data(results_dir, config_folder, metrics=None):
    metrics = metrics or METRICS
    folder_path = os.path.join(results_dir, config_folder)
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    avg_records = []
    std_records = []

    for csv_file in csv_files:
        model_name = extract_model_name(os.path.basename(csv_file), config_folder)
        if model_name is None:
            continue

        df = pd.read_csv(csv_file, index_col=0)
        if df.empty:
            continue

        row = df.iloc[0]
        avg_row = {"model": model_name}
        std_row = {"model": model_name}

        for metric in metrics:
            if metric not in row.index:
                continue
            avg_val, std_val = split_avg_std(row[metric])
            avg_row[metric] = avg_val
            std_row[metric] = std_val

        avg_records.append(avg_row)
        std_records.append(std_row)

    if not avg_records:
        return pd.DataFrame(columns=metrics), pd.DataFrame(columns=metrics)

    avg_df = pd.DataFrame(avg_records).set_index("model")
    std_df = pd.DataFrame(std_records).set_index("model")
    return avg_df, std_df


def short_config_name(config_name):
    short_name = re.sub(r"^Severity_Max_", "", config_name)
    short_name = re.sub(r"_Trails=.*$", "", short_name)
    return short_name


def infer_smote_flag(config_name):
    return "True" if "SMOTE=True" in config_name else "False"


def infer_optimization_metric(config_name):
    if "balanced_accuracy" in config_name:
        return "balanced_accuracy"
    if "f1_weighted" in config_name:
        return "f1_weighted"
    if "mcc" in config_name:
        return "mcc"
    return "accuracy"


def manifest_path(results_dir):
    return os.path.join(results_dir, "experiment_manifest.json")


def save_manifest(manifest):
    path = manifest["manifest_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def require_manifest(results_dir, explicit_path=None):
    path = os.path.abspath(explicit_path or manifest_path(results_dir))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run run_experiment.py first to generate a complete workspace."
        )
    manifest = load_manifest(path)
    manifest["manifest_path"] = path
    return manifest


def validate_single_row_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        raise ValueError(f"Empty result file: {csv_path}")
    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one result row in {csv_path}, found {len(df)}. "
            "Use a single target combination for the portable pipeline."
        )
    return df


def validate_complete_results(manifest):
    errors = []
    expected_models = manifest["expected_models"]

    for config_meta in manifest["expected_configs"]:
        config_name = config_meta["config_name"]
        summary_dir = config_meta["summary_dir"]
        if not os.path.isdir(summary_dir):
            errors.append(f"Missing summary directory: {summary_dir}")
        else:
            for model_name in expected_models:
                csv_path = result_csv_path(summary_dir, config_name, model_name)
                if not os.path.isfile(csv_path):
                    errors.append(f"Missing summary file: {csv_path}")
                    continue
                try:
                    validate_single_row_csv(csv_path)
                except ValueError as exc:
                    errors.append(str(exc))

        for run_dir in config_meta["run_dirs"]:
            if not os.path.isdir(run_dir):
                errors.append(f"Missing run directory: {run_dir}")
                continue
            for model_name in expected_models:
                csv_path = result_csv_path(run_dir, config_name, model_name)
                if not os.path.isfile(csv_path):
                    errors.append(f"Missing run file: {csv_path}")
                    continue
                try:
                    validate_single_row_csv(csv_path)
                except ValueError as exc:
                    errors.append(str(exc))

    if errors:
        preview = "\n".join(errors[:20])
        extra = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more"
        raise RuntimeError(
            "Result set is incomplete. VoteNRank and RGI require every expected run "
            f"and summary file.\n{preview}{extra}"
        )

    return True
