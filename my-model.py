"""my-model.py
Your best model. Add your features in load_data(), write your model in train_and_predict(), then run
iterations 1 to 5 from the project folder:

    python my-model.py --iteration 1      # ... up to --iteration 5

Feel free to create any files, classes or functions you need for your model (a file you import
needs a name without '-', e.g. networks.py). Call them from train_and_predict(), so all training
stays inside ResourceLogger, and keep the rest of this structure.

Outputs go to analysis/my-best-model-analysis/. For a task, copy this file to task-<N>.py and
change OUTPUT_DIR (and MODEL_NAME for Task 1, e.g. "my-model-size100").
example-for-guide/run-example.py is a complete, working version of the same steps.
"""
import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "helper-src"))  # "helper-src" has a hyphen, so it is added to the path
from loading import load_csv_files  # noqa: E402
from splitting import BASE_SEED, stratified_split  # noqa: E402
from logger import ResourceLogger  # noqa: E402
from outputs import output_paths, capture_output, save_test_results, combine_iteration_metrics  # noqa: E402

DATA_DIR = "/home/shared"
ID_COLUMN = "Id"
TARGET_COLUMN = "binary_label"
OUTPUT_DIR = Path("analysis/my-best-model-analysis")
MODEL_NAME = "my-model"

# Chronic pain (binary_label = 1): at least one of the patient's conditions has one of these words
# in its description. These are the 31 keywords shown in the project explorer.
PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]


def load_data():
    """One row per patient from patients.csv, with binary_label built from conditions.csv."""
    tables = load_csv_files(DATA_DIR, ["patients", "conditions"])
    patients, conditions = tables["patients"], tables["conditions"]

    pattern = r"\b(?:" + "|".join(map(re.escape, PAIN_KEYWORDS)) + r")\b"
    is_pain = conditions["DESCRIPTION"].str.contains(pattern, case=False, na=False, regex=True)
    patients[TARGET_COLUMN] = patients[ID_COLUMN].isin(conditions.loc[is_pain, "PATIENT"]).astype(int)

    # Add your features here, e.g. from other CSV files in DATA_DIR (observations, medications, ...).
    # Do not use the condition descriptions above as features: they define the label.
    return patients


def train_and_predict(train_df, val_df, test_df, seed):
    """Train on train_df, choose settings with val_df, and return P(class 1) for every row of test_df.

    Use test_df only to predict: never train or tune on it, or read its binary_label.
    """
    raise NotImplementedError("Write your model in train_and_predict()")


def main(iteration):
    os.chdir(PROJECT_DIR)
    seed = BASE_SEED + iteration  # 43 to 47 for iterations 1 to 5
    paths = output_paths(OUTPUT_DIR, MODEL_NAME, iteration)

    with capture_output(paths["log"]):  # everything printed is also saved to the .out file
        print(f"Model: {MODEL_NAME} | iteration: {iteration} | seed: {seed}")
        df = load_data()
        print(f"{len(df):,} patients | {df[TARGET_COLUMN].mean():.1%} chronic pain")
        splits = stratified_split(df, TARGET_COLUMN, ID_COLUMN, seed=seed)
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

        with ResourceLogger(tag=f"{MODEL_NAME}-iter{iteration}", log_path=paths["resource_usage"]) as usage:
            test_prob = train_and_predict(train_df, val_df, test_df, seed)  # all training happens in here

        save_test_results(paths, test_df[TARGET_COLUMN], test_prob, iteration, seed, usage)
    combine_iteration_metrics(OUTPUT_DIR, MODEL_NAME)  # updates my-model-metrics-all-iterations.csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My best model for the final project.")
    parser.add_argument("--iteration", type=int, default=1, help="1 to 5 (seed = 42 + iteration)")
    args = parser.parse_args()
    if args.iteration < 1:
        parser.error("--iteration must be 1 or more")
    main(args.iteration)
