# AutoResearcher — Scientific Benchmark Report

## 1. Executive Summary & Hypothesis Verification

> **Technical Hypothesis:** Does an autonomous ML agent that selects its next experiment based on rich validation error diagnostics (train-val overfitting gap, per-class F1 breakdown, confusion matrix pairs) discover higher-performing models in fewer trials than Random Search, Grid Search, and black-box Bayesian Optimization (Optuna TPE)?

### Quantitative Comparison by Dataset

| Dataset | Strategy | Baseline F1 | Best Val Macro-F1 (Mean ± Std) | Gain over Baseline | Locked Holdout Test F1 | Trials to 95% Peak | Convergence AUC | Mean Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `credit-g` | **AutoResearcher (Agent)** | 0.8858 | **0.9070** ± 0.0000 | +0.0212 | **0.9230** ± 0.0000 | 4.0 | 0.8986 | 14.7s |
| `credit-g` | Grid Search | 0.8858 | **0.8919** ± 0.0000 | +0.0061 | **0.8746** ± 0.0000 | 1.0 | 0.8912 | 8.4s |
| `credit-g` | Optuna (TPE) | 0.8858 | **0.9042** ± 0.0000 | +0.0184 | **0.8983** ± 0.0000 | 7.0 | 0.9001 | 35.0s |
| `credit-g` | Random Search | 0.8858 | **0.9189** ± 0.0000 | +0.0331 | **0.9094** ± 0.0000 | 3.0 | 0.9073 | 46.1s |
| `phoneme` | **AutoResearcher (Agent)** | 0.7854 | **0.8121** ± 0.0000 | +0.0267 | **0.7957** ± 0.0000 | 7.0 | 0.8025 | 21.9s |
| `phoneme` | Grid Search | 0.7854 | **0.8014** ± 0.0000 | +0.0160 | **0.8249** ± 0.0000 | 5.0 | 0.7924 | 8.4s |
| `phoneme` | Optuna (TPE) | 0.7854 | **0.8088** ± 0.0000 | +0.0234 | **0.8332** ± 0.0000 | 1.0 | 0.8059 | 29.3s |
| `phoneme` | Random Search | 0.7854 | **0.8090** ± 0.0000 | +0.0236 | **0.8080** ± 0.0000 | 2.0 | 0.8031 | 35.2s |
| `vehicle` | **AutoResearcher (Agent)** | 0.8537 | **0.8904** ± 0.0000 | +0.0368 | **0.9013** ± 0.0000 | 2.0 | 0.8812 | 226.4s |
| `vehicle` | Grid Search | 0.8537 | **0.8594** ± 0.0000 | +0.0057 | **0.8901** ± 0.0000 | 5.0 | 0.8572 | 14.3s |
| `vehicle` | Optuna (TPE) | 0.8537 | **0.8725** ± 0.0000 | +0.0188 | **0.9010** ± 0.0000 | 7.0 | 0.8627 | 467.2s |
| `vehicle` | Random Search | 0.8537 | **0.8537** ± 0.0000 | +0.0000 | **0.8657** ± 0.0000 | 0.0 | 0.8537 | 121.9s |

---

## 2. Sample Agent Scientific Lab Notebook (Key Breakthrough Decisions)

### Dataset: `credit-g` (Best Val F1: `0.9070` | Holdout F1: `0.9230`)

- **Trial #0** (`Val Macro-F1 = 0.8858`, `Overfit Gap = +0.1090`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'credit-g' (800 dev rows, 16 numeric & 0 categorical features, imbalance_ratio=2.25). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Changes Applied:* `baseline_initialization`
- **Trial #3** (`Val Macro-F1 = 0.9031`, `Overfit Gap = +0.0932`, `Model = xgboost`):
  - *Diagnosis:* Best validation score is 0.8858 with worst class 'bad' (F1=0.8380). Depth-wise gradient boosting (`xgboost`) with engineered interactions has not been tested.
  - *Hypothesis:* If we test `xgboost` (depth=5, lr=0.08, reg_lambda=3.0) combined with `ratios_top5` feature interactions and `robust` scaling, non-linear boundaries between confused classes will sharpen.
  - *Changes Applied:* `scaler: none -> robust, feature_engineering: none -> ratios_top5, imbalance_strategy: none -> class_weight_balanced, model_family: random_forest -> xgboost, n_estimators: 100 -> 250, max_depth: 10 -> 5, learning_rate: 0.1 -> 0.08, regularization: 1.0 -> 3.0, subsample: 1.0 -> 0.85, min_samples_leaf: 1 -> 2`
- **Trial #4** (`Val Macro-F1 = 0.9070`, `Overfit Gap = +0.0890`, `Model = xgboost`):
  - *Diagnosis:* Representation bottleneck on Trial #3 (Val F1=0.9031, top confusion: 'bad' misclassified as 'good' (34 errors, 13.8%)).
  - *Hypothesis:* If we apply `ratios_top5` feature engineering with `standard` scaling while keeping the winning `xgboost` backbone, Macro-F1 will improve because: Non-linear ratio and difference interactions among top-5 F-value features capture relative magnitudes between confused classes.
  - *Changes Applied:* `scaler: robust -> standard`

### Dataset: `phoneme` (Best Val F1: `0.8121` | Holdout F1: `0.7957`)

- **Trial #0** (`Val Macro-F1 = 0.7854`, `Overfit Gap = +0.2091`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'phoneme' (960 dev rows, 8 numeric & 0 categorical features, imbalance_ratio=1.08). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Changes Applied:* `baseline_initialization`
- **Trial #2** (`Val Macro-F1 = 0.7975`, `Overfit Gap = +0.2017`, `Model = catboost`):
  - *Diagnosis:* Current best uses 'random_forest' (Val F1=0.7854), leaving symmetric oblivious trees (`catboost`) unexplored. Top confusion is 'oral' misclassified as 'nasal' (118 errors, 25.5%).
  - *Hypothesis:* If we switch to `catboost` (250 iterations, depth=6, lr=0.08, L2=3.0) with `onehot` encoding, symmetric decision tables will reduce variance and better separate overlapping decision boundaries.
  - *Changes Applied:* `model_family: random_forest -> catboost, n_estimators: 100 -> 250, max_depth: 10 -> 6, learning_rate: 0.1 -> 0.08, regularization: 1.0 -> 3.0, subsample: 1.0 -> 0.85`
- **Trial #3** (`Val Macro-F1 = 0.8099`, `Overfit Gap = +0.1387`, `Model = catboost`):
  - *Diagnosis:* High variance / overfitting detected in Trial #2: Train Macro-F1=0.9992 vs Val Macro-F1=0.7975 (overfit_gap=+0.2017).
  - *Hypothesis:* If we constrain tree complexity (`max_depth=4`, `min_samples_leaf=4`), apply stochastic subsampling (`subsample=0.8`), and raise L2 regularization to 12.0, generalization on validation folds will improve.
  - *Changes Applied:* `max_depth: 6 -> 4, regularization: 3.0 -> 12.0, subsample: 0.85 -> 0.8, min_samples_leaf: 1 -> 4`
- **Trial #7** (`Val Macro-F1 = 0.8121`, `Overfit Gap = +0.1488`, `Model = catboost`):
  - *Diagnosis:* Representation bottleneck on Trial #3 (Val F1=0.8099, top confusion: 'oral' misclassified as 'nasal' (95 errors, 20.6%)).
  - *Hypothesis:* If we apply `poly_top5` feature engineering with `robust` scaling while keeping the winning `catboost` backbone, Macro-F1 will improve because: Second-order polynomial terms (`x_i * x_j`, `x_i^2`) on top informative columns increase boundary expressiveness.
  - *Changes Applied:* `scaler: none -> robust, feature_engineering: none -> poly_top5`

### Dataset: `vehicle` (Best Val F1: `0.8904` | Holdout F1: `0.9013`)

- **Trial #0** (`Val Macro-F1 = 0.8537`, `Overfit Gap = +0.1452`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'vehicle' (676 dev rows, 18 numeric & 0 categorical features, imbalance_ratio=1.03). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Changes Applied:* `baseline_initialization`
- **Trial #2** (`Val Macro-F1 = 0.8904`, `Overfit Gap = +0.0969`, `Model = catboost`):
  - *Diagnosis:* Current best uses 'random_forest' (Val F1=0.8537), leaving symmetric oblivious trees (`catboost`) unexplored. Top confusion is 'opel' misclassified as 'van' (15 errors, 8.8%).
  - *Hypothesis:* If we switch to `catboost` (250 iterations, depth=6, lr=0.08, L2=3.0) with `onehot` encoding, symmetric decision tables will reduce variance and better separate overlapping decision boundaries.
  - *Changes Applied:* `model_family: random_forest -> catboost, n_estimators: 100 -> 250, max_depth: 10 -> 6, learning_rate: 0.1 -> 0.08, regularization: 1.0 -> 3.0, subsample: 1.0 -> 0.85`
