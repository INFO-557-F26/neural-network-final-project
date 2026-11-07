# neural-network-final-project

INFO 557 final project: predict **chronic pain** (`binary_label`) from synthetic
[Synthea](https://synthea.mitre.org/) patient records.

- **Instructions:** [`my-model-instructions.ipynb`](my-model-instructions.ipynb)
- **Complete example:** [`example-for-guide/`](example-for-guide/)
- **Tasks and benchmarks:** [project explorer](https://ai-sandbox-ai-healthcare.github.io/neural-network-final-project/).
  Put your task results in your report only, but commit all their output files in `analysis/`.
- **`my_model.csv`:** download the template from the explorer (Deliverables tab), fill it in with your best
  model's results for the 5 iterations, and upload it back to the explorer to complete the comparison.
- **The explorer counts toward your grade:** it records how long you use it, what you do there, and whether
  your model beats the baseline and the GPU models and reaches the final goal.
- **Data:** CSV files in the shared folder `/home/shared/` on Cacao JupyterHub (do not copy them into the
  project). This example loads `patients.csv`; change the file name to load another CSV file:

  ```python
  df = pd.read_csv("/home/shared/patients.csv")
  ```

## Steps

1. Log in to Cacao JupyterHub with your GitHub account.
2. Create a GitHub access token (see the project instructions document).
3. Clone this repo in JupyterHub.
4. Create and activate a Python environment in your JupyterHub space:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
5. In the environment: `pip install -r requirements.txt`
6. Run the example: `python example-for-guide/run-example.py`
7. Explore the `.py` files in `helper-src/`.
8. Read the instructions: `my-model-instructions.ipynb`.
9. Complete `my-model.py`.
10. Create a `.py` file for each of your other models (`task-1.py` ... `task-5.py`).
11. Commit and push your code and output files after each part: the best model and each task (see below).

## Folders

```
neural-network-final-project/
├── my-model-instructions.ipynb     instructions and helper examples
├── my-model.py                     your best model (starter)
├── task-1.py ... task-5.py         your tasks (copies of my-model.py)
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

**Commit and push after each part:** at least one commit for your best model and one for each task.
One commit at the end of the project costs a letter grade.

```bash
git add *.py analysis/
git commit -m "Task 1"   # one commit per part: best model, Task 1, ..., Task 5
git push
```

- **Push:** your `.py` files (including any you created) and the output files in `analysis/`, including
  `analysis/overall-resource-usage-summary.csv`. Missing files will affect your score.
- **Do not push:** data, model weights, or changes to `helper-src/` and
  `example-for-guide/`.
