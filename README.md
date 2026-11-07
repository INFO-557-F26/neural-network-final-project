# neural-network-final-project

INFO 557 final project: predict **chronic pain** (`binary_label`) from synthetic
[Synthea](https://synthea.mitre.org/) patient records.

- **[Project explorer](https://ai-sandbox-ai-healthcare.github.io/neural-network-final-project/):** the tasks,
  the benchmarks, and where you upload your best model. **It counts toward your grade:** it records how long
  you use it, what you do there, and whether your model beats the baseline and the GPU models and reaches the
  final goal.
- **Instructions:** [`my-model-instructions.ipynb`](my-model-instructions.ipynb)
- **Complete example:** [`example-for-guide/`](example-for-guide/)

## Steps

1. **Explore:** open the project explorer and read the Deliverables tab: the main task, Tasks 1 to 5, and the
   benchmarks.
2. **Set up:** log in to Cacao JupyterHub with your GitHub account, clone this repo, and install the requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Learn the code:** run `python example-for-guide/run-example.py`, look through `helper-src/`, and read
   `my-model-instructions.ipynb`.
4. **Best model:** complete `my-model.py` and run iterations 1 to 5. Commit and push.
5. **Back to the explorer:** download the `my_model.csv` template (Deliverables tab), fill it in with your best
   model's 5 iterations, and upload it. Goal: beat the baseline and the GPU models and reach the final goal.
6. **Tasks 1 to 5:** copy `my-model.py` to `task-1.py` ... `task-5.py` and change what each task asks.
   Commit and push each task as you finish it. Task results go in your report only, not the explorer.
7. **Report:** submit it to D2L.

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

**Push as you go:** complete each part (best model, Task 1 ... Task 5) and push it separately. Several commits
per task are encouraged. **Pushing everything in one change at the end costs a letter grade.**

```bash
git add *.py analysis/
git commit -m "Task 1: training sizes"   # one or more commits per task
git push
```

- **Push:** your `.py` files (including any you created) and the output files in `analysis/`, including
  `analysis/overall-resource-usage-summary.csv`. Missing files will affect your score.
- **Do not push:** data, model weights, or changes to `helper-src/` and `example-for-guide/`.
