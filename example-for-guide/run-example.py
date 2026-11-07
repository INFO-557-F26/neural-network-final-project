"""run-example.py
Complete example: the same problem and the same dataset as Problem 1 (Assignment 1), solved
with the Assignment 1 logistic regression and the helpers in helper-src/.

Steps: load -> stratified 70/15/15 split -> standardize (train only) -> train, watching the
validation loss (timed by ResourceLogger) -> validation metrics -> one test evaluation -> save files.

Output files (iteration N) in example-for-guide/:
  results/metrics/logistic-regression-metrics-iterN.csv
  results/figures/logistic-regression-confusion-matrix-iterN.png
  results/figures/logistic-regression-roc-curve-iterN.png
  logs/logistic-regression-iterN.out
  logs/resource-usage.csv
Every run also adds a row to analysis/overall-resource-usage-summary.csv.

Run it from the project folder:
    python example-for-guide/run-example.py                 # iteration 1 (seed 43)
    python example-for-guide/run-example.py --iteration 2   # iteration 2 (seed 44), and so on

Your own starter is my-model.py in the project folder; it follows the same steps.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "helper-src"))  # "helper-src" has a hyphen, so it is added to the path
from loading import check_required_columns  # noqa: E402
from splitting import BASE_SEED, stratified_split, validate_split  # noqa: E402
from logger import ResourceLogger  # noqa: E402
from evaluation import compute_metrics  # noqa: E402
from outputs import output_paths, capture_output, save_test_results  # noqa: E402

# --------------------------------------------------------------
# Settings (paths are relative to the project folder)
# --------------------------------------------------------------
OUTPUT_DIR = Path("example-for-guide")
DATA_FILE = OUTPUT_DIR / "data" / "patient-features.csv"
MODEL_NAME = "logistic-regression"  # your project uses "my-model"

ID_COLUMN = "id"
TARGET_COLUMN = "binary_label"
FEATURE_COLUMNS = [  # the 19 numeric features used in Assignment 1 (condition_text is left out: it defines the label)
    "age", "is_female", "number_of_unique_meds", "number_of_encounters",
    "number_of_procedures", "unique_procedures", "pain_severity",
    "body_height", "body_weight", "body_mass_index",
    "systolic_blood_pressure", "diastolic_blood_pressure",
    "heart_rate", "respiratory_rate",
    "qaly", "daly", "qols", "healthcare_expenses", "healthcare_coverage",
]
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Default hyperparameters. In your project, choose these with the validation split.
LEARNING_RATE = 0.1
STEPS = 500
THRESHOLD = 0.5   # predict class 1 when the probability is >= THRESHOLD
LOG_EVERY = 100   # print the train/validation loss every LOG_EVERY steps


# --------------------------------------------------------------
# Logistic regression from Assignment 1 (NumPy only)
# --------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def standardize_features(feature_matrix):
    """Z-score each column. Returns (scaled_matrix, means, stds); call it on TRAINING data only."""
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (feature_matrix - means) / stds, means, stds


def predict_proba(feature_matrix, weights, bias):
    return sigmoid(feature_matrix.dot(weights) + bias)


def logistic_regression_gradients(feature_matrix, y, weights, bias):
    error = predict_proba(feature_matrix, weights, bias) - y
    return feature_matrix.T.dot(error) / len(y), float(np.mean(error))


def log_loss(y, probs, eps=1e-9):
    return float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))


def train_logistic_regression(x_train, y_train, iterations, learning_rate, x_val, y_val, log_every):
    """Gradient descent from all-zero weights; prints the train and validation loss as it goes."""
    weights, bias = np.zeros(x_train.shape[1]), 0.0
    for step in range(1, iterations + 1):
        grad_weights, grad_bias = logistic_regression_gradients(x_train, y_train, weights, bias)
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
        if step % log_every == 0 or step == iterations:
            train_loss = log_loss(y_train, predict_proba(x_train, weights, bias))
            val_loss = log_loss(y_val, predict_proba(x_val, weights, bias))
            print(f"  step {step:>4}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}")
    return weights, bias


# --------------------------------------------------------------
# One iteration
# --------------------------------------------------------------
def main(iteration):
    os.chdir(PROJECT_DIR)  # paths in the settings are relative to the project folder
    seed = BASE_SEED + iteration  # 43 to 47 for iterations 1 to 5
    paths = output_paths(OUTPUT_DIR, MODEL_NAME, iteration)

    with capture_output(paths["log"]):  # everything printed below also goes to the .out file
        print(f"Model: {MODEL_NAME} | iteration: {iteration} | seed: {seed}")
        print(f"Hyperparameters: learning_rate={LEARNING_RATE}, steps={STEPS}, threshold={THRESHOLD}")
        print(f"Split fractions: {SPLIT_FRACTIONS}")

        # 1. Load the data
        df = pd.read_csv(DATA_FILE)
        check_required_columns(df, [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS], DATA_FILE.name)
        print(f"\n[1] Loaded {DATA_FILE.as_posix()}: {len(df)} patients, {len(FEATURE_COLUMNS)} features, "
              f"{df[TARGET_COLUMN].mean():.1%} positive")

        # 2. Stratified train / validation / test split
        print("\n[2] Stratified split")
        splits = stratified_split(df, TARGET_COLUMN, ID_COLUMN, SPLIT_FRACTIONS, seed)
        print(validate_split(splits, df, TARGET_COLUMN, ID_COLUMN, SPLIT_FRACTIONS).to_string())
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        y_train, y_val, y_test = (part[TARGET_COLUMN].to_numpy(dtype=float) for part in (train_df, val_df, test_df))

        # 3-4. Features and training, timed by the resource logger (all training goes inside this block)
        print("\n[3] Training (timed by ResourceLogger)")
        with ResourceLogger(tag=f"{MODEL_NAME}-iter{iteration}", log_path=paths["resource_usage"]) as usage:
            x_train, means, stds = standardize_features(train_df[FEATURE_COLUMNS].to_numpy(dtype=float))
            x_val = (val_df[FEATURE_COLUMNS].to_numpy(dtype=float) - means) / stds  # train statistics only
            weights, bias = train_logistic_regression(
                x_train, y_train, STEPS, LEARNING_RATE, x_val, y_val, LOG_EVERY
            )
            val_prob = predict_proba(x_val, weights, bias)

        print("\nLearned parameters (features were standardized, so weights are comparable):")
        learned = pd.DataFrame({"feature": FEATURE_COLUMNS, "weight": weights})
        learned = learned.reindex(learned["weight"].abs().sort_values(ascending=False).index)
        print(learned.to_string(index=False, float_format="%.4f"))
        print(f"bias: {bias:.4f}")

        # 5. Validation metrics: use these to compare settings, not as the final result
        print("\n[4] Validation metrics")
        for name, value in compute_metrics(y_val, val_prob, THRESHOLD).items():
            print(f"  {name:<12} {value:.4f}")

        # 6. Test: evaluated once, after every choice is made; then save the output files
        print("\n[5] Test evaluation and output files")
        x_test = (test_df[FEATURE_COLUMNS].to_numpy(dtype=float) - means) / stds
        test_prob = predict_proba(x_test, weights, bias)
        save_test_results(paths, y_test, test_prob, iteration, seed, usage, THRESHOLD)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic regression example for the final project.")
    parser.add_argument("--iteration", type=int, default=1, help="1, 2, 3, ... (seed = 42 + iteration)")
    args = parser.parse_args()
    if args.iteration < 1:
        parser.error("--iteration must be 1 or more")
    main(args.iteration)
