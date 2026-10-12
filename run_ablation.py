import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_generation import generate_dataset, extract_features
from run_experiments import split_groups, evaluate_probabilities, set_seed


FEATURE_NAMES = [
    "mean",
    "variance",
    "skewness",
    "kurtosis",
    "mean_correlation",
    "temporal_difference",
    "periodicity",
    "max_excursion",
]

MODELS = ("SVM_RBF", "Random_Forest")


def build_model(name, seed):
    if name == "SVM_RBF":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=0.8,
                gamma="scale",
                kernel="rbf",
                probability=True,
                random_state=seed,
            ),
        )

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=seed,
        n_jobs=-1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="EchoForensics temporal-difference feature ablation"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55]
    )
    parser.add_argument("--output-dir", default="outputs_ablation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in args.seeds:
        print(f"\n=== Ablation seed {seed} ===")
        set_seed(seed)
        X, y, groups = generate_dataset(seed)
        split = split_groups(groups, seed)

        features = extract_features(X)

        # Original eight-feature baseline versus seven features without
        # temporal_difference (feature index 5).
        feature_sets = {
            "all_8_features": np.arange(8),
            "without_temporal_difference": np.array(
                [i for i in range(8) if i != 5]
            ),
        }

        for feature_set_name, columns in feature_sets.items():
            for model_name in MODELS:
                model = build_model(model_name, seed)
                model.fit(
                    features[split["train"]][:, columns],
                    y[split["train"]],
                )

                probabilities = model.predict_proba(
                    features[split["test"]][:, columns]
                )

                classes = (
                    model[-1].classes_
                    if hasattr(model, "steps")
                    else model.classes_
                )

                aligned = np.zeros(
                    (len(probabilities), 4), dtype=float
                )
                for j, class_id in enumerate(classes):
                    aligned[:, int(class_id)] = probabilities[:, j]

                metrics, predictions = evaluate_probabilities(
                    y[split["test"]], aligned
                )

                row = {
                    "seed": seed,
                    "feature_set": feature_set_name,
                    "features_used": len(columns),
                    "model": model_name,
                    **metrics,
                }
                rows.append(row)

                cm = confusion_matrix(
                    y[split["test"]],
                    predictions,
                    labels=np.arange(4),
                )
                replay_recall = (
                    cm[3, 3] / cm[3].sum()
                    if cm[3].sum()
                    else float("nan")
                )

                print(
                    f"{feature_set_name:32s} "
                    f"{model_name:14s} "
                    f"accuracy={metrics['accuracy']:.4f} "
                    f"macro_F1={metrics['macro_f1']:.4f} "
                    f"Replay_recall={replay_recall:.4f}"
                )
                rows[-1]["replay_spoof_recall"] = replay_recall

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "ablation_by_seed.csv", index=False)

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_ovr_auc",
        "replay_spoof_recall",
    ]
    summary = (
        results.groupby(["feature_set", "model"])[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary.reset_index().to_csv(
        out_dir / "ablation_summary.csv", index=False
    )

    print(f"\nSaved results to: {out_dir.resolve()}")
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
