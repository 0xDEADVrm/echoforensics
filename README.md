# EchoForensics — reproducible synthetic benchmark

This is a **reference implementation**, not recovered source code for the original manuscript.
It generates a synthetic CSI-like benchmark and evaluates SVM, Random Forest, and a
CNN–BiLSTM–Attention model. The resulting metrics may differ from the manuscript's
unverified figures. Do not report results until the script has completed successfully.

## Important scientific scope

- Inputs are simulated channel-amplitude arrays, not measurements collected from Wi-Fi hardware.
- The benchmark has four labels: `Empty`, `Benign`, `Intrusion`, and `Replay/Spoof`.
- Each scene group contains one example of every class. All four examples from a scene
  stay in the same split. This avoids scene-level train/test leakage and makes class
  counts balanced, but it is a controlled synthetic benchmark rather than a field study.
- Replay samples are derived from the Empty or Benign example in the same scene group.
- The SHA-256 ledger test is a software integrity demonstration, not a complete
  implementation of a production evidence custody system.
- Noise robustness uses additive noise on the synthetic amplitude input. It is not
  a calibrated physical-channel SNR study.
- No claim of real-world detection, hardware challenge-response efficacy, or deployment
  security follows from these synthetic experiments.

## 1. Create a virtual environment (Windows PowerShell)

```powershell
cd echoforensics_reference
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install a Python version supported by your installed
PyTorch build and use `python -m venv .venv`.

## 2. Run a quick smoke test

```powershell
python run_experiments.py --seeds 7 --epochs 3 --output-dir outputs_smoke
```

This verifies the pipeline but is **not** the final reported experiment.

## 3. Run the experiment

For one reproducible seed:

```powershell
python run_experiments.py --seeds 42 --epochs 30 --output-dir outputs_seed42
```

For a small multi-seed run:

```powershell
python run_experiments.py --seeds 11 22 33 44 55 --epochs 30 --output-dir outputs_multiseed
```

Training can be slow on CPU. To use a CUDA-enabled PyTorch installation, the script
automatically selects CUDA when available. Keep the exact command, Python/PyTorch
versions, device, and output files with the research records.

## Outputs

- `metrics_by_seed.csv` — per-seed test metrics for all three models.
- `summary_mean_std.csv` — mean and standard deviation across seeds.
- `confusion_matrix_primary.csv` — CNN confusion matrix for the first seed.
- `per_class_auc_primary.csv` — CNN one-vs-rest AUC per class for the first seed.
- `classification_report_primary.csv` — per-class precision/recall/F1/support.
- `robustness_primary.csv` — CNN accuracy after additive synthetic noise.
- `split_manifest_primary.csv` — scene group and split assignment.
- `integrity_test_primary.json` — SHA-256 chain tamper test output.
- `run_metadata.json` — settings, package versions, and explicit scope notes.
- `training_history_primary.csv` — CNN training/validation history.
- `primary_test_predictions.csv` — CNN test labels and predicted probabilities.

## What to update in the paper

Use only values actually present in these outputs. Update the abstract, tables, discussion,
and conclusion to match. Do not retain the manuscript's 98.2%, 0.999 AUC, baseline metrics,
noise figures, or exact split counts unless a fresh run reproduces them. Describe the data
as synthetic and avoid implying physical validation. Cite and audit all references separately.

The group split here is intentionally different from the draft's claimed 1530/270/600 split:
600 scene groups × four classes are split as 382/68/150 groups, yielding 1528/272/600
windows. If the final code is changed, re-run all experiments and keep the exact code version.
