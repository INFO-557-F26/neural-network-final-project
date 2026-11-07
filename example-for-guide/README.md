# Example: logistic regression

The same problem and the same dataset as Problem 1 (Assignment 1), solved with the
Assignment 1 logistic regression and the helpers in `helper-src/`. Use it as a guide.

## Run

From the project folder:

```bash
python example-for-guide/run-example.py                 # iteration 1 (seed 43)
python example-for-guide/run-example.py --iteration 2   # iteration 2 (seed 44)
```

It loads the data, makes a stratified 70/15/15 split, trains on train, checks validation,
evaluates once on test, and saves:

```
example-for-guide/
├── run-example.py
├── data/patient-features.csv
├── logs/
│   ├── logistic-regression-iter1.out              everything the run printed
│   └── resource-usage.csv                         time, GPU, CPU per run
└── results/
    ├── metrics/logistic-regression-metrics-iter1.csv
    └── figures/
        ├── logistic-regression-confusion-matrix-iter1.png
        └── logistic-regression-roc-curve-iter1.png
```

Each run also adds a row to `analysis/overall-resource-usage-summary.csv`.

## Your project

`my-model.py` in the project folder follows the same steps: write your model in
`train_and_predict()`. Its outputs go to `analysis/my-best-model-analysis/` with the same
names, starting with `my-model-` instead of `logistic-regression-`.

Do not push changes to this folder.
