from __future__ import annotations
import argparse, json, os, platform, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import CLASS_NAMES, ExperimentConfig, N_GROUPS
from data_generation import generate_dataset, extract_features
from models import CNNBiLSTMAttention
from integrity import run_tamper_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism can reduce performance; this makes runs more reproducible.
    torch.use_deterministic_algorithms(True, warn_only=True)

def split_groups(groups, seed):
    unique = np.unique(groups)
    # Fixed counts by whole scene groups: 382 train, 68 validation, 150 test.
    rng = np.random.default_rng(seed + 9001)
    shuffled = rng.permutation(unique)
    train_g, val_g, test_g = shuffled[:382], shuffled[382:450], shuffled[450:]
    assert len(train_g) == 382 and len(val_g) == 68 and len(test_g) == 150
    assert not (set(train_g) & set(val_g) or set(train_g) & set(test_g) or set(val_g) & set(test_g))
    idx = {
        "train": np.flatnonzero(np.isin(groups, train_g)),
        "val": np.flatnonzero(np.isin(groups, val_g)),
        "test": np.flatnonzero(np.isin(groups, test_g)),
    }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(groups[idx[a]]) .isdisjoint(set(groups[idx[b]])), f"group leakage: {a}/{b}"
    return idx

def normalize_cnn(X, train_idx):
    # Per-subcarrier z-score, using training data only.
    mu = X[train_idx].mean(axis=(0, 2), keepdims=True)
    sd = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6
    return ((X - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)

def evaluate_probabilities(y_true, prob):
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=np.arange(len(CLASS_NAMES)), average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, prob, multi_class="ovr", average="macro",
                            labels=np.arange(len(CLASS_NAMES)))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(p), "macro_recall": float(r),
        "macro_f1": float(f1), "macro_ovr_auc": float(auc)
    }, pred

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train_cnn(X, y, idx, cfg, device, output_dir):
    Xn, mu, sd = normalize_cnn(X, idx["train"])
    train_loader = make_loader(Xn[idx["train"]], y[idx["train"]], cfg.batch_size, True)
    val_loader = make_loader(Xn[idx["val"]], y[idx["val"]], cfg.batch_size, False)
    model = CNNBiLSTMAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses, val_true, val_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(float(criterion(logits, yb).cpu()))
                val_pred.extend(logits.argmax(1).cpu().numpy().tolist())
                val_true.extend(yb.cpu().numpy().tolist())
        val_loss = float(np.mean(val_losses))
        val_acc = float(accuracy_score(val_true, val_pred))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"    epoch {epoch:02d}/{cfg.epochs}: train_loss={np.mean(losses):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print("    early stopping")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    def predict_indices(indices, array=Xn):
        probs = []
        loader = make_loader(array[indices], y[indices], cfg.batch_size, False)
        with torch.no_grad():
            for xb, _ in loader:
                logits = model(xb.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)
    return model, Xn, mu, sd, predict_indices, pd.DataFrame(history)

def run_one_seed(seed, cfg, out_dir, primary=False):
    print(f"\n=== Seed {seed} ===")
    set_seed(seed)
    X, y, groups = generate_dataset(seed)
    idx = split_groups(groups, seed)
    split_names = np.full(len(y), "", dtype=object)
    for name, inds in idx.items():
        split_names[inds] = name
    # Split counts and no-leakage checks.
    split_rows = pd.DataFrame({"sample_index": np.arange(len(y)), "group_id": groups,
                               "label": [CLASS_NAMES[i] for i in y], "split": split_names})
    for name, inds in idx.items():
        counts = np.bincount(y[inds], minlength=4)
        print(f"  {name}: {len(inds)} windows; class counts={counts.tolist()}")
        assert np.all(counts > 0)
    split_sets = [set(groups[idx[k]]) for k in ("train", "val", "test")]
    assert split_sets[0].isdisjoint(split_sets[1]) and split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])

    # Classical models: features extracted per sample, scaler fit on train only.
    features = extract_features(X)
    classical = {
        "SVM_RBF": make_pipeline(StandardScaler(), SVC(C=0.8, gamma="scale", kernel="rbf",
                                                       probability=True, random_state=seed)),
        "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                 random_state=seed, n_jobs=-1)
    }
    rows = []
    for name, model in classical.items():
        model.fit(features[idx["train"]], y[idx["train"]])
        prob = model.predict_proba(features[idx["test"]])
        # Ensure columns align with class order.
        if hasattr(model, "classes_"):
            classes = model.classes_
        else:
            classes = model[-1].classes_
        aligned = np.zeros((len(prob), len(CLASS_NAMES)), dtype=float)
        for j, c in enumerate(classes):
            aligned[:, int(c)] = prob[:, j]
        metrics, _ = evaluate_probabilities(y[idx["test"]], aligned)
        rows.append({"seed": seed, "model": name, **metrics})
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  CNN device: {device}")
    model, Xn, mu, sd, predict, history = train_cnn(X, y, idx, cfg, device, out_dir)
    prob = predict(idx["test"])
    cnn_metrics, pred = evaluate_probabilities(y[idx["test"]], prob)
    rows.append({"seed": seed, "model": "CNN_BiLSTM_Attention", **cnn_metrics})
    print("  CNN_BiLSTM_Attention: " + ", ".join(f"{k}={v:.4f}" for k, v in cnn_metrics.items()))

    if primary:
        out_dir.mkdir(parents=True, exist_ok=True)
        split_rows.to_csv(out_dir / "split_manifest_primary.csv", index=False)
        history.to_csv(out_dir / "training_history_primary.csv", index=False)
        cm = confusion_matrix(y[idx["test"]], pred, labels=np.arange(4))
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "confusion_matrix_primary.csv")
        report = classification_report(y[idx["test"]], pred, labels=np.arange(4),
                                       target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report_primary.csv")
        auc_rows = []
        from sklearn.preprocessing import label_binarize
        ybin = label_binarize(y[idx["test"]], classes=np.arange(4))
        for j, name in enumerate(CLASS_NAMES):
            auc_rows.append({"class": name, "one_vs_rest_auc": float(roc_auc_score(ybin[:, j], prob[:, j]))})
        pd.DataFrame(auc_rows).to_csv(out_dir / "per_class_auc_primary.csv", index=False)
        pred_df = pd.DataFrame({"sample_index": idx["test"], "true_label": [CLASS_NAMES[i] for i in y[idx["test"]]],
                                "predicted_label": [CLASS_NAMES[i] for i in pred]})
        for j, name in enumerate(CLASS_NAMES):
            pred_df[f"prob_{name.replace('/', '_')}"] = prob[:, j]
        pred_df.to_csv(out_dir / "primary_test_predictions.csv", index=False)

        # Additive-noise robustness, using train-derived normalization only.
        robustness = []
        rng = np.random.default_rng(seed + 123456)
        for sigma in cfg.noise_levels:
            noisy = X[idx["test"]] + rng.normal(0, sigma, X[idx["test"]].shape).astype(np.float32)
            noisy_n = ((noisy - mu) / sd).astype(np.float32)
            noisy_prob_parts = []
            loader = make_loader(noisy_n, y[idx["test"]], cfg.batch_size, False)
            model.eval()
            with torch.no_grad():
                for xb, _ in loader:
                    noisy_prob_parts.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
            noisy_prob = np.vstack(noisy_prob_parts)
            nm, _ = evaluate_probabilities(y[idx["test"]], noisy_prob)
            robustness.append({"noise_sigma": sigma, **nm})
        pd.DataFrame(robustness).to_csv(out_dir / "robustness_primary.csv", index=False)
        with open(out_dir / "integrity_test_primary.json", "w", encoding="utf-8") as f:
            json.dump(run_tamper_test(), f, indent=2)
        print("  Primary-seed detailed outputs saved.")
    return rows

def main():
    parser = argparse.ArgumentParser(description="Run EchoForensics synthetic reference benchmark.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="One or more integer seeds.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not args.seeds:
        parser.error("epochs and batch-size must be positive, and at least one seed is required.")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(epochs=args.epochs, batch_size=args.batch_size)
    all_rows = []
    start = time.time()
    for i, seed in enumerate(args.seeds):
        rows = run_one_seed(seed, cfg, out_dir, primary=(i == 0))
        all_rows.extend(rows)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    summary = metrics.groupby("model").agg(["mean", "std"])
    # Flatten multi-index columns for easy insertion into paper tables.
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    metadata = {
        "seeds": args.seeds, "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "n_groups": N_GROUPS, "windows_per_group": 4, "total_windows": N_GROUPS*4,
        "split_groups": {"train": 382, "validation": 68, "test": 150},
        "split_windows": {"train": 1528, "validation": 272, "test": 600},
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": round(time.time()-start, 2),
        "scope_warning": "Synthetic-only benchmark; no physical Wi-Fi CSI collection or hardware validation."
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("\n=== Mean ± standard deviation across seeds ===")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")
    print("Do not copy manuscript metrics unless they match these actual outputs.")

if __name__ == "__main__":
    main()
