import argparse
import os

import pandas as pd

from analysis_common import (
    default_results_dir,
    load_config_data,
    require_manifest,
    short_config_name,
    validate_complete_results,
)
from votenrank import Leaderboard


def parse_args():
    parser = argparse.ArgumentParser(description="Run VoteNRank on saved model result tables.")
    parser.add_argument(
        "--results-dir",
        default=default_results_dir(),
        help="Directory containing one subfolder per configuration.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for VoteNRank CSV outputs. Defaults to results-dir.",
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


def build_big_table(results_dir, config_folders):
    tables = []
    for config_folder in config_folders:
        avg_df, _ = load_config_data(results_dir, config_folder)
        if avg_df.empty:
            continue
        renamed = avg_df.copy()
        prefix = short_config_name(config_folder)
        renamed.columns = [f"{prefix}__{column}" for column in renamed.columns]
        tables.append(renamed)

    if not tables:
        raise ValueError("No valid configuration tables were loaded.")

    big_table = pd.concat(tables, axis=1)
    if big_table.isna().any().any():
        big_table = big_table.fillna(big_table.mean(numeric_only=True))
    return big_table


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir or results_dir)
    manifest = require_manifest(results_dir, args.manifest)
    validate_complete_results(manifest)

    config_folders = args.configs or [cfg["config_name"] for cfg in manifest["expected_configs"]]

    os.makedirs(out_dir, exist_ok=True)

    big_table = build_big_table(results_dir, config_folders)
    leaderboard = Leaderboard(table=big_table)

    winners = leaderboard.elect_all()
    full_ranking = leaderboard.rank_all()

    winners_path = os.path.join(out_dir, "votenrank_winners.csv")
    ranking_path = os.path.join(out_dir, "votenrank_full_ranking.csv")
    winners.to_csv(winners_path, index=False)
    full_ranking.to_csv(ranking_path)

    print(f"Loaded {len(config_folders)} configurations and {big_table.shape[0]} models.")
    print(f"Saved {winners_path}")
    print(f"Saved {ranking_path}")


if __name__ == "__main__":
    main()
