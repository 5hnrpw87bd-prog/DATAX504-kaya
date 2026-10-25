# Corporate Bankruptcy Risk Prediction

Cost-sensitive binary classification on the UCI Taiwanese Bankruptcy Prediction
dataset. The project compares a standardised logistic-regression baseline with a
compact Keras multilayer perceptron (MLP) trained with and without balanced class
weights, then evaluates decision thresholds under hypothetical
false-negative:false-positive cost ratios of 2:1, 5:1 and 10:1.

This is a historical benchmark study, not a model validated for current lending,
credit or investment decisions.

## Results at a glance

| Model | Validation AP | Test AP | Test ROC-AUC |
|---|---:|---:|---:|
| No-skill | 0.0323 | 0.0323 | 0.5000 |
| Logistic regression | 0.2587 | 0.2943 | 0.9015 |
| MLP, unweighted | 0.2730 | 0.3135 | 0.8160 |
| MLP, class-weighted | **0.2819** | **0.3145** | 0.8569 |

The class-weighted MLP had the highest validation AP (0.2819) and was selected
before the test set was examined. Its test AP advantage over the unweighted MLP
is only 0.0010, so the evidence that class weighting improves AP is limited.
Logistic regression has the highest test ROC-AUC (0.9015), illustrating that
model rankings depend on the evaluation metric.

Validation-selected thresholds are approximately 0.8948, 0.7350 and 0.6771 for
the 2:1, 5:1 and 10:1 cost ratios. At 5:1, the selected threshold gives 16 true
positives, 40 false positives and 17 false negatives (precision 28.57%, recall
48.48%); the illustrative cost is 125 versus 264 at the default 0.5 threshold.
At 2:1, the selected rule costs 61 versus 66 for issuing no alerts while missing
27 of 33 bankrupt observations, so its practical advantage is small.

## Data

- Source: [UCI Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction), DOI [10.24432/C5004D](https://doi.org/10.24432/C5004D).
- 6,819 observations, 95 financial predictors, 220 bankrupt cases (3.23%).
- Taiwanese firms, 1999–2009; licensed CC BY 4.0.
- No missing values, exact duplicate rows or non-numeric columns.
- No usable row-level time index is available, so the evaluation uses held-out
  observations rather than a later historical period.

A copy of `data/data.csv` is included (about 11 MB). If the file is absent, the
notebook downloads and caches it from UCI automatically.

## Environment

- Python 3.12
- TensorFlow 2.x, Keras 3.15.0
- scikit-learn 1.9.0, NumPy 2.5.1, pandas 3.0.3
- matplotlib and joblib
- CPU is sufficient; no GPU is required

These are the versions used for the recorded run. `results/experiment_config.json`
records the Keras, scikit-learn, NumPy and pandas versions saved by the notebook.
The repository-level `requirements.txt` uses minimum versions (`>=`); use the
versions above if exact reproduction matters.

## Setup

From the course repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

## Run the experiment

1. Open `final_project/bankruptcy_prediction.ipynb` in JupyterLab.
2. Restart the kernel and run all cells in order.

Jupyter uses the notebook directory as the working directory, so the notebook
writes to `final_project/data/`, `final_project/models/` and
`final_project/results/`. A full run takes a few minutes on CPU.

## Project structure

```text
final_project/
├── bankruptcy_prediction.ipynb
├── README.md
├── data/
│   └── data.csv
├── models/
│   ├── mlp_weighted.keras
│   ├── scaler.joblib
│   └── logistic_model.joblib
└── results/
    ├── experiment_config.json
    ├── validation_metrics.csv
    ├── test_metrics.csv
    ├── validation_thresholds.csv
    ├── test_thresholds.csv
    ├── test_predictions.csv
    ├── split_rows.csv
    ├── extreme_input_diagnostic.csv
    └── input_sensitivity.csv
```

## Experimental design

- Fixed random seed: 42.
- Stratified 70% / 15% / 15% train / validation / test split:
  4,773 / 1,023 / 1,023 observations, with 154 / 33 / 33 bankrupt cases.
- `StandardScaler` fitted on training data only; the same transformation is
  applied to validation and test data.
- Class weights calculated from training labels only:
  0.5167 for non-bankrupt and 15.4968 for bankrupt observations (ratio ≈ 30:1).
- Model selection uses validation AP only; the test set is examined after the
  model is fixed.
- EarlyStopping monitors Keras validation PR-AUC, with patience 10 and
  `restore_best_weights=True`.
- Thresholds for 2:1, 5:1 and 10:1 false-negative:false-positive costs are
  selected on validation data and applied unchanged to test data.
- Primary metric: Average Precision (AP), computed with
  `sklearn.metrics.average_precision_score`. ROC-AUC is secondary. Accuracy is
  not used because a constant "not bankrupt" prediction would already achieve
  about 96.77%.
- Keras PR-AUC and scikit-learn AP use different calculations. EarlyStopping
  uses the former for monitoring; all formal model comparisons use AP.

## Outputs

- `models/mlp_weighted.keras`: selected Keras model.
- `models/scaler.joblib`: training-fitted scaler.
- `models/logistic_model.joblib`: logistic-regression baseline.
- `results/`: validation and test metrics, threshold tables, per-row test
  predictions, split assignments and the extreme-input diagnostic.

## Reproduction check

The notebook:

- checks that no original row appears in more than one split;
- saves the exact split assignments, predictions and metric tables;
- reloads `mlp_weighted.keras` and `scaler.joblib` and verifies that validation
  predictions are reproduced with a maximum absolute difference of 0.0.

Exact retraining can still vary slightly across hardware, TensorFlow versions or
random-number implementations. The saved artifacts and split file allow the
reported results to be checked without retraining.

## Limitations

- Historical Taiwanese data (1999–2009); concept drift and cross-market
  generalisation cannot be evaluated without newer data.
- Each validation and test set contains only 33 bankrupt cases. Results are
  based on one split and one seed, without confidence intervals or repeated-run
  analysis.
- Model selection and threshold selection use the same validation set; with
  only 33 positive cases, both choices may be sensitive to individual
  observations.
- The 2:1, 5:1 and 10:1 costs are hypothetical scenario units, not measured
  financial losses.
- Sigmoid outputs are model scores, not calibrated bankruptcy probabilities.
- Observation 379 has an extreme "Revenue per person" value (7.05 × 10⁹) far
  outside the training range. The sensitivity check is diagnostic only; it does
  not change the reported test results.

## AI assistance

ChatGPT (Sol 6.0) and Codex (OpenAI) assisted with step-by-step code suggestions,
explanations of machine-learning concepts, interpretation of outputs, diagnostic
suggestions and drafting documentation and report text.

I was responsible for the research questions, evaluation protocol, threshold
choices, interpretation of the results and the final text. I entered and executed
the code locally, inspected the outputs and completed the reproduction check.
The reported metrics and diagnostic findings come from the locally executed
experiment. The final submitted wording and my understanding of the
implementation remain my responsibility.

## Report

The final report is submitted separately as `fp_final_Meng_Kaya.pdf`.
