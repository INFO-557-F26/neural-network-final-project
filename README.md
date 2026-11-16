# neural-network-final-project

INFO 557 final project: predict **chronic pain** (`binary_label`) from synthetic
[Synthea](https://synthea.mitre.org/) patient records.

- **[Project explorer](https://ai-sandbox-ai-healthcare.github.io/neural-network-final-project/):** the tasks,
  the benchmarks, and where you upload your best model. **It counts toward your grade:** it records how long
  you use it, what you do there, and whether your model beats the baseline and the GPU models and reaches the
  final goal.
- **Instructions:** [`my-model-instructions.ipynb`](my-model-instructions.ipynb)
- **Complete example:** [`example-for-guide/`](example-for-guide/)

## Set up

1. Open the [D2L final-project guide](https://d2l.arizona.edu/d2l/le/lessons/1799446/topics/21331990), then accept the [Classroom 50 assignment](https://classroom50.org/INFO-557-F26/ista-457-info-557-neural-networks/assignments/final-project/accept).
2. Use your personal Classroom 50 repository. Copy its HTTPS URL from **Code**, then clone it in a Cacao JupyterHub terminal and enter the repository folder. See D2L for course JupyterHub access details. Verify that you can read `/home/shared/patients.csv` and `/home/shared/conditions.csv`.
3. Create and activate your environment from the repository root.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

## Run and upload your best model

Run `python example-for-guide/run-example.py` and inspect its logs and results. The example uses its bundled sample. The full project uses the shared course dataset. Read [the instructions notebook](my-model-instructions.ipynb) for helper examples.

Complete `load_data()` and `train_and_predict()` in `my-model.py`. Fit preprocessing and the model on training data, tune with validation data, and use test data only for final scoring. Keep label-defining descriptions, diagnosis codes that reveal the label, and equivalent proxies out of your predictors. Return each test patient’s probability of class 1 in test-row order.

~~~bash
for i in 1 2 3 4 5; do python my-model.py --iteration "$i" || break; done
~~~

The iterations use seeds 43–47. Confirm that `analysis/my-best-model-analysis/results/metrics/my-model-metrics-all-iterations.csv` has five completed rows. In **Deliverables (Tasks & Submission)** in the [project explorer](https://ai-sandbox-ai-healthcare.github.io/neural-network-final-project/), download the `my_model.csv` template and copy in those rows with the original headers. Upload the CSV and confirm **My Model** updates. The upload is saved in the current browser. Compare positive-class F1 and compute cost with the CPU and GPU benchmarks.

## Complete Tasks 1–5

Use the best model as the reference. Use `task-1.py`, `task-2.py`, `task-3.py`, and `task-5.py` for the experiments, with separate `analysis/task-N-analysis/` folders. Commit any code you write for Task 4. Put your findings in the report.

1. **Training data size.** For each seed, take stratified samples of 100, 1,000, and 4,000 from that seed’s main-task training set. Keep the corresponding validation and test patients fixed across sizes. Reuse the full-size main-task run for each seed (8,646 training patients). Report the mean and standard deviation for each size.
2. **Feature removal.** Rank features on training or validation data. Record what you expect, remove the top two features, and rerun the same five splits. Compare the mean and standard deviation with your main-task results.
3. **Missing data.** Restore the full feature set. Use a fixed masking seed to hide half of each of the top three features in each split. Fit imputation on training data and apply it to all three splits. Rerun and summarize the five main-task splits.
4. **Leakage audit.** Review the 16 raw CSVs for direct and indirect leakage, compare common conditions and comorbidities across classes, trace one diagnosis across related tables, and audit your predictors. This task requires no model runs.
5. **Better label.** Define and justify revised criteria, including how they handle comorbidity. Report the number of changed labels and the new class balance. Reuse the original train, validation, and test patient IDs for each seed, then attach the revised labels. Retrain five times and interpret the scores for the revised target.

## Submit your report and prepare for peer review

Use **Submit | Final Project** in D2L. Include your best model and feature choices, performance and cost comparisons, what you expected and observed for every task, unsuccessful experiments, and the exact commands and settings needed to reproduce your results. Include the GitHub commit link and expected results for the required peer reproducibility review. Run the notebook’s `check_outputs()` and `check_budget()` checks before submitting.

## Folders

```
neural-network-final-project/
├── my-model-instructions.ipynb     instructions and helper examples
├── my-model.py                     your best model (starter)
├── task-1.py ... task-5.py         your tasks (Task 4 is a written audit)
├── helper-src/                     helpers (do not edit)
├── example-for-guide/              complete example (do not edit)
└── analysis/
    ├── overall-resource-usage-summary.csv   every run (written for you)
    ├── my-best-model-analysis/     logs/, results/metrics/, results/figures/
    ├── task-1-analysis/            same layout
    ├── ...
    └── task-5-analysis/
```

Output files are named `my-model-<what>-iter<N>.<extension>`, e.g. `my-model-metrics-iter1.csv`, in the
folder of the model or task. For Task 1, add the training size: `my-model-size100-metrics-iter1.csv`.

## Compute budget: 4 hours

The budget is the run time of your code inside `ResourceLogger`, not the time you are logged in or working
on JupyterHub. Every run counts, including reruns and failed runs. Each run is also recorded in the course
resource sheet with your JupyterHub username.

| Total time | Budget points |
|---|---|
| up to 4 h | full credit |
| 4 to 5 h | full credit, flagged |
| 5 to 8 h | lost in proportion |
| over 8 h | all lost |

**Modifying `overall-resource-usage-summary.csv` on purpose is a violation of academic integrity.**

## Push to GitHub

**Push as you go:** complete each part (best model, Task 1 ... Task 5) and push it separately. Several commits
per task are encouraged. **Pushing everything in one change at the end costs a letter grade.**

```bash
git add *.py analysis/
git commit -m "Task 1: training sizes"   # one or more commits per task
git push
```

- **Push:** your `.py` files (including any you created) and the output files in `analysis/`, including
  `analysis/overall-resource-usage-summary.csv`. Missing files will affect your score.
- **Do not push:** data, model weights, `my_model.csv`, or changes to `helper-src/` and `example-for-guide/`.
