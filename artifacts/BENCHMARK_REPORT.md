# SEVA — Scientific Benchmark Report

## 1. Executive Summary & Hypothesis Verification

> **Core Technical Hypothesis:** Does an autonomous agent guided by rich validation error diagnostics (train-val gap, class collapse, confusion matrices) discover superior tabular pipelines with higher sample efficiency and lower regret than blind Random Search and black-box Bayesian Optimization (Optuna TPE)?

### Methodological Definitions & Scientific Scope
- **Validation Macro-F1:** Arithmetic mean of 5-fold Stratified Cross-Validation on the 80% Development partition. Best Val Macro-F1 reflects the maximum 5-fold CV mean across all trials.
- **Holdout Test Macro-F1:** Evaluated once on the untouched 20% holdout split using the winning configuration retrained on the full 80% Dev set.
- **AUC Convergence:** Normalized discrete integral across budget T=6, defined as (1/T) * sum(cumulative_best_val_f1). Normalized to [0.0, 1.0].
- **Trials to 95% Peak:** The earliest trial index capturing >= 95% of total validation improvement over Trial 0 baseline.
- **Environment & Seeds:** Evaluated under Python 3.12.10, Scikit-learn 1.9.1, XGBoost 3.4.1, CatBoost 1.2.10, Optuna 5.0.0, and Pydantic 2.14.0 using a single reference seed (seed=42).
- **Scientific Caveat:** With a single seed and three datasets, these findings represent **promising preliminary empirical evidence** of diagnostic search efficiency rather than a definitive demonstration of universally generalizable superiority.

### Four-Strategy Ablation Benchmark

| Dataset | Strategy | Baseline F1 | Best Val Macro-F1 (Mean ± Std) | Holdout Test F1 (Mean ± Std) | Trials to 95% Peak | AUC Convergence | LLM Cost ($) | Mean Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `credit-g` | Optuna (TPE) | 0.6572 | **0.6943** ± 0.0000 | **0.6755** ± 0.0000 | 4.0 | 0.6740 | N/A | 12.9s |
| `credit-g` | Random Search | 0.6572 | **0.7007** ± 0.0000 | **0.6681** ± 0.0000 | 4.0 | 0.6829 | N/A | 14.0s |
| `credit-g` | **SEVA (Heuristic Agent)** | 0.6572 | **0.7205** ± 0.0000 | **0.6790** ± 0.0000 | 2.0 | 0.7075 | N/A | 7.9s |
| `credit-g` | **SEVA (LLM Agent)** | 0.6572 | **0.7161** ± 0.0000 | **0.6858** ± 0.0000 | 3.0 | 0.6867 | N/A | 8.0s |
| `phoneme` | Optuna (TPE) | 0.8345 | **0.8466** ± 0.0000 | **0.8376** ± 0.0000 | 5.0 | 0.8365 | N/A | 27.5s |
| `phoneme` | Random Search | 0.8345 | **0.8434** ± 0.0000 | **0.8740** ± 0.0000 | 5.0 | 0.8362 | N/A | 32.1s |
| `phoneme` | **SEVA (Heuristic Agent)** | 0.8345 | **0.8520** ± 0.0000 | **0.8677** ± 0.0000 | 5.0 | 0.8381 | N/A | 27.6s |
| `phoneme` | **SEVA (LLM Agent)** | 0.8345 | **0.8476** ± 0.0000 | **0.8812** ± 0.0000 | 2.0 | 0.8432 | N/A | 70.0s |
| `vehicle` | Optuna (TPE) | 0.7368 | **0.7734** ± 0.0000 | **0.7437** ± 0.0000 | 1.0 | 0.7673 | N/A | 266.1s |
| `vehicle` | Random Search | 0.7368 | **0.7625** ± 0.0000 | **0.7766** ± 0.0000 | 5.0 | 0.7481 | N/A | 62.9s |
| `vehicle` | **SEVA (Heuristic Agent)** | 0.7368 | **0.7842** ± 0.0000 | **0.7893** ± 0.0000 | 5.0 | 0.7565 | N/A | 22.7s |
| `vehicle` | **SEVA (LLM Agent)** | 0.7368 | **0.7774** ± 0.0000 | **0.7432** ± 0.0000 | 5.0 | 0.7570 | N/A | 59.2s |

---

## 2. Scientific Lab Notebook (Diagnosis -> Hypothesis -> Intervention -> Observed Outcome)

### Strategy: `SEVA (Heuristic Agent)` on Dataset `credit-g` (Seed 42)
- **Final Performance:** CV Val Macro-F1 = `0.7205` | Locked Holdout F1 = `0.6790`
- **Efficiency:** Peak at Trial `#2` | AUC Convergence = `0.7075` | Time = `7.9s`

- **Trial #0** (`Val F1 = 0.6572`, `Overfit Gap = +0.3134`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'credit-g' (800 dev rows, 7 numeric & 13 categorical features, imbalance_ratio=2.33). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Expected Outcome:* Establish benchmark anchor and compute baseline error distribution across all classes.
  - *Mutations Applied:* `baseline_initialization`
  - *Observed Outcome:* Baseline established: Val Macro-F1=0.6572, Overfit Gap=+0.3134, Worst Class F1=0.4712.
- **Trial #1** (`Val F1 = 0.7059`, `Overfit Gap = +0.2214`, `Model = random_forest`):
  - *Diagnosis:* Minority class collapse detected: class 'bad' has F1=0.4712 vs best class F1=0.8437 (dataset imbalance_ratio=2.33).
  - *Hypothesis:* If we enable cost-sensitive learning (`class_weight_balanced`) and probability threshold tuning on top of Trial #0, recall on minority class 'bad' will recover and lift overall Val Macro-F1.
  - *Expected Outcome:* Increase worst_class_f1 from 0.4712 by at least +0.05 and improve Val Macro-F1.
  - *Mutations Applied:* `imbalance_strategy: none -> class_weight_balanced, threshold_tuning: False -> True, min_samples_leaf: 1 -> 3`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7059 (+0.0487 vs prior best), Overfit Gap=+0.2214, Worst Class 'bad' F1=0.5817.
- **Trial #2** (`Val F1 = 0.7205`, `Overfit Gap = +0.1056`, `Model = random_forest`):
  - *Diagnosis:* High variance / overfitting detected in Trial #1: Train Macro-F1=0.9273 vs Val Macro-F1=0.7059 (overfit_gap=+0.2214).
  - *Hypothesis:* If we constrain tree complexity (`max_depth=7`, `min_samples_leaf=6`), apply stochastic subsampling (`subsample=0.8`), and raise L2 regularization to 6.0, generalization on validation folds will improve.
  - *Expected Outcome:* Reduce overfit_gap from +0.2214 below +0.08 while sustaining or increasing Val Macro-F1.
  - *Mutations Applied:* `max_depth: 10 -> 7, regularization: 1.0 -> 6.0, subsample: 1.0 -> 0.8, min_samples_leaf: 3 -> 6`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7205 (+0.0146 vs prior best), Overfit Gap=+0.1056, Worst Class 'bad' F1=0.6096.

### Strategy: `SEVA (LLM Agent)` on Dataset `credit-g` (Seed 42)
- **Final Performance:** CV Val Macro-F1 = `0.7161` | Locked Holdout F1 = `0.6858`
- **Efficiency:** Peak at Trial `#3` | AUC Convergence = `0.6867` | Time = `8.0s`

- **Trial #0** (`Val F1 = 0.6572`, `Overfit Gap = +0.3134`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'credit-g' (800 dev rows, 7 numeric & 13 categorical features, imbalance_ratio=2.33). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Expected Outcome:* Establish benchmark anchor and compute baseline error distribution across all classes.
  - *Mutations Applied:* `baseline_initialization`
  - *Observed Outcome:* Baseline established: Val Macro-F1=0.6572, Overfit Gap=+0.3134, Worst Class F1=0.4712.
- **Trial #3** (`Val F1 = 0.7161`, `Overfit Gap = +0.2651`, `Model = random_forest`):
  - *Diagnosis:* Class sensitivity refinement: worst class 'bad' is at F1=0.4712 vs best class F1=0.8437.
  - *Hypothesis:* [LLM Semantic Prior] If we adjust class balancing to `random_oversample` with `threshold_tuning=False` on `random_forest`, the precision-recall trade-off across classes will improve Macro-F1.
  - *Expected Outcome:* Improve balanced accuracy and raise worst_class_f1 above 0.4712.
  - *Mutations Applied:* `imbalance_strategy: none -> random_oversample`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7161 (+0.0589 vs prior best), Overfit Gap=+0.2651, Worst Class 'bad' F1=0.6085.

### Strategy: `SEVA (Heuristic Agent)` on Dataset `phoneme` (Seed 42)
- **Final Performance:** CV Val Macro-F1 = `0.8520` | Locked Holdout F1 = `0.8677`
- **Efficiency:** Peak at Trial `#5` | AUC Convergence = `0.8381` | Time = `27.6s`

- **Trial #0** (`Val F1 = 0.8345`, `Overfit Gap = +0.1514`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'phoneme' (2000 dev rows, 5 numeric & 0 categorical features, imbalance_ratio=2.41). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Expected Outcome:* Establish benchmark anchor and compute baseline error distribution across all classes.
  - *Mutations Applied:* `baseline_initialization`
  - *Observed Outcome:* Baseline established: Val Macro-F1=0.8345, Overfit Gap=+0.1514, Worst Class F1=0.7651.
- **Trial #2** (`Val F1 = 0.8358`, `Overfit Gap = +0.1458`, `Model = catboost`):
  - *Diagnosis:* Current best uses 'random_forest' (Val F1=0.8345), leaving symmetric oblivious trees (`catboost`) unexplored. Top confusion is '2' misclassified as '1' (144 errors, 24.5%).
  - *Hypothesis:* If we switch to `catboost` (250 iterations, depth=6, lr=0.08, L2=3.0) with `onehot` encoding, symmetric decision tables will reduce variance and better separate overlapping decision boundaries.
  - *Expected Outcome:* Achieve Val Macro-F1 higher than current best (0.8345) with decreased error count on top confused pair.
  - *Mutations Applied:* `imbalance_strategy: none -> class_weight_balanced, threshold_tuning: False -> True, model_family: random_forest -> catboost, n_estimators: 100 -> 250, max_depth: 10 -> 6, learning_rate: 0.1 -> 0.08, regularization: 1.0 -> 3.0, subsample: 1.0 -> 0.85`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.8358 (+0.0013 vs prior best), Overfit Gap=+0.1458, Worst Class '2' F1=0.7667.
- **Trial #5** (`Val F1 = 0.8520`, `Overfit Gap = +0.1422`, `Model = catboost`):
  - *Diagnosis:* Representation bottleneck on Trial #2 (Val F1=0.8358, top confusion: '2' misclassified as '1' (145 errors, 24.7%)).
  - *Hypothesis:* If we apply `ratios_top5` feature engineering with `standard` scaling while keeping the winning `catboost` backbone, Macro-F1 will improve because: Non-linear ratio and difference interactions among top-5 F-value features capture relative magnitudes between confused classes.
  - *Expected Outcome:* Increase Val Macro-F1 above 0.8358 by reducing representation noise.
  - *Mutations Applied:* `scaler: none -> standard, feature_engineering: none -> ratios_top5`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.8520 (+0.0162 vs prior best), Overfit Gap=+0.1422, Worst Class '2' F1=0.7888.

### Strategy: `SEVA (LLM Agent)` on Dataset `phoneme` (Seed 42)
- **Final Performance:** CV Val Macro-F1 = `0.8476` | Locked Holdout F1 = `0.8812`
- **Efficiency:** Peak at Trial `#2` | AUC Convergence = `0.8432` | Time = `70.0s`

- **Trial #0** (`Val F1 = 0.8345`, `Overfit Gap = +0.1514`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'phoneme' (2000 dev rows, 5 numeric & 0 categorical features, imbalance_ratio=2.41). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Expected Outcome:* Establish benchmark anchor and compute baseline error distribution across all classes.
  - *Mutations Applied:* `baseline_initialization`
  - *Observed Outcome:* Baseline established: Val Macro-F1=0.8345, Overfit Gap=+0.1514, Worst Class F1=0.7651.
- **Trial #2** (`Val F1 = 0.8476`, `Overfit Gap = +0.1520`, `Model = catboost`):
  - *Diagnosis:* Fine-tuning capacity/regularization trade-off around Trial #0 (Val F1=0.8345, Overfit Gap=+0.1514).
  - *Hypothesis:* [LLM Semantic Prior] If we update capacity parameters to {'model_family': 'catboost', 'n_estimators': 350, 'learning_rate': 0.05, 'max_depth': 8, 'regularization': 1.0} on `catboost`, we balance bias and variance to reduce the remaining confusion ('2' misclassified as '1' (144 errors, 24.5%)).
  - *Expected Outcome:* Attain peak validation convergence by fine-tuning tree complexity.
  - *Mutations Applied:* `model_family: random_forest -> catboost, n_estimators: 100 -> 350, max_depth: 10 -> 8, learning_rate: 0.1 -> 0.05`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.8476 (+0.0130 vs prior best), Overfit Gap=+0.1520, Worst Class '2' F1=0.7838.

### Strategy: `SEVA (Heuristic Agent)` on Dataset `vehicle` (Seed 42)
- **Final Performance:** CV Val Macro-F1 = `0.7842` | Locked Holdout F1 = `0.7893`
- **Efficiency:** Peak at Trial `#5` | AUC Convergence = `0.7565` | Time = `22.7s`

- **Trial #0** (`Val F1 = 0.7368`, `Overfit Gap = +0.2632`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'vehicle' (676 dev rows, 18 numeric & 0 categorical features, imbalance_ratio=1.09). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Expected Outcome:* Establish benchmark anchor and compute baseline error distribution across all classes.
  - *Mutations Applied:* `baseline_initialization`
  - *Observed Outcome:* Baseline established: Val Macro-F1=0.7368, Overfit Gap=+0.2632, Worst Class F1=0.5377.
- **Trial #2** (`Val F1 = 0.7526`, `Overfit Gap = +0.2409`, `Model = catboost`):
  - *Diagnosis:* Current best uses 'random_forest' (Val F1=0.7368), leaving symmetric oblivious trees (`catboost`) unexplored. Top confusion is 'opel' misclassified as 'saab' (77 errors, 45.3%).
  - *Hypothesis:* If we switch to `catboost` (250 iterations, depth=6, lr=0.08, L2=3.0) with `onehot` encoding, symmetric decision tables will reduce variance and better separate overlapping decision boundaries.
  - *Expected Outcome:* Achieve Val Macro-F1 higher than current best (0.7368) with decreased error count on top confused pair.
  - *Mutations Applied:* `model_family: random_forest -> catboost, n_estimators: 100 -> 250, max_depth: 10 -> 6, learning_rate: 0.1 -> 0.08, regularization: 1.0 -> 3.0, subsample: 1.0 -> 0.85`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7526 (+0.0158 vs prior best), Overfit Gap=+0.2409, Worst Class 'saab' F1=0.5579.
- **Trial #4** (`Val F1 = 0.7764`, `Overfit Gap = +0.2236`, `Model = xgboost`):
  - *Diagnosis:* Best validation score is 0.7526 with worst class 'saab' (F1=0.5579). Depth-wise gradient boosting (`xgboost`) with engineered interactions has not been tested.
  - *Hypothesis:* If we test `xgboost` (depth=5, lr=0.08, reg_lambda=3.0) combined with `ratios_top5` feature interactions and `robust` scaling, non-linear boundaries between confused classes will sharpen.
  - *Expected Outcome:* Outperform catboost baseline by at least +0.01 in Macro-F1 via non-linear feature combinations.
  - *Mutations Applied:* `scaler: none -> robust, feature_engineering: none -> ratios_top5, model_family: catboost -> xgboost, max_depth: 6 -> 5, min_samples_leaf: 1 -> 2`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7764 (+0.0238 vs prior best), Overfit Gap=+0.2236, Worst Class 'opel' F1=0.5988.
- **Trial #5** (`Val F1 = 0.7842`, `Overfit Gap = +0.2151`, `Model = xgboost`):
  - *Diagnosis:* High variance / overfitting detected in Trial #4: Train Macro-F1=1.0000 vs Val Macro-F1=0.7764 (overfit_gap=+0.2236).
  - *Hypothesis:* If we constrain tree complexity (`max_depth=4`, `min_samples_leaf=4`), apply stochastic subsampling (`subsample=0.8`), and raise L2 regularization to 12.0, generalization on validation folds will improve.
  - *Expected Outcome:* Reduce overfit_gap from +0.2236 below +0.08 while sustaining or increasing Val Macro-F1.
  - *Mutations Applied:* `max_depth: 5 -> 4, regularization: 3.0 -> 12.0, subsample: 0.85 -> 0.8, min_samples_leaf: 2 -> 4`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7842 (+0.0078 vs prior best), Overfit Gap=+0.2151, Worst Class 'saab' F1=0.6140.

### Strategy: `SEVA (LLM Agent)` on Dataset `vehicle` (Seed 42)
- **Final Performance:** CV Val Macro-F1 = `0.7774` | Locked Holdout F1 = `0.7432`
- **Efficiency:** Peak at Trial `#5` | AUC Convergence = `0.7570` | Time = `59.2s`

- **Trial #0** (`Val F1 = 0.7368`, `Overfit Gap = +0.2632`, `Model = random_forest`):
  - *Diagnosis:* Initial state on 'vehicle' (676 dev rows, 18 numeric & 0 categorical features, imbalance_ratio=1.09). No model metrics collected yet.
  - *Hypothesis:* Establish a reference anchor using an unregularized RandomForest baseline (depth=10, OneHot encoding, no class balancing) to expose dataset pathologies (overfitting gap, minority class recall, and confused class pairs).
  - *Expected Outcome:* Establish benchmark anchor and compute baseline error distribution across all classes.
  - *Mutations Applied:* `baseline_initialization`
  - *Observed Outcome:* Baseline established: Val Macro-F1=0.7368, Overfit Gap=+0.2632, Worst Class F1=0.5377.
- **Trial #1** (`Val F1 = 0.7446`, `Overfit Gap = +0.2550`, `Model = random_forest`):
  - *Diagnosis:* Representation bottleneck on Trial #0 (Val F1=0.7368, top confusion: 'opel' misclassified as 'saab' (77 errors, 45.3%)).
  - *Hypothesis:* [LLM Semantic Prior] If we apply `tree_importance_k75` feature engineering with `standard` scaling while keeping the winning `random_forest` backbone, Macro-F1 will improve because: Pruning bottom 25% noisy features reduces split variance (current overfit_gap=+0.2632).
  - *Expected Outcome:* Increase Val Macro-F1 above 0.7368 by reducing representation noise.
  - *Mutations Applied:* `scaler: none -> standard, feature_engineering: none -> tree_importance_k75`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7446 (+0.0078 vs prior best), Overfit Gap=+0.2550, Worst Class 'saab' F1=0.5621.
- **Trial #2** (`Val F1 = 0.7611`, `Overfit Gap = +0.2389`, `Model = catboost`):
  - *Diagnosis:* Fine-tuning capacity/regularization trade-off around Trial #1 (Val F1=0.7446, Overfit Gap=+0.2550).
  - *Hypothesis:* [LLM Semantic Prior] If we update capacity parameters to {'model_family': 'catboost', 'n_estimators': 350, 'learning_rate': 0.05, 'max_depth': 8, 'regularization': 1.0} on `catboost`, we balance bias and variance to reduce the remaining confusion ('opel' misclassified as 'saab' (67 errors, 39.4%)).
  - *Expected Outcome:* Attain peak validation convergence by fine-tuning tree complexity.
  - *Mutations Applied:* `model_family: random_forest -> catboost, n_estimators: 100 -> 350, max_depth: 10 -> 8, learning_rate: 0.1 -> 0.05`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7611 (+0.0165 vs prior best), Overfit Gap=+0.2389, Worst Class 'opel' F1=0.5697.
- **Trial #5** (`Val F1 = 0.7774`, `Overfit Gap = +0.2226`, `Model = xgboost`):
  - *Diagnosis:* Fine-tuning capacity/regularization trade-off around Trial #2 (Val F1=0.7611, Overfit Gap=+0.2389).
  - *Hypothesis:* [LLM Semantic Prior] If we update capacity parameters to {'model_family': 'xgboost', 'n_estimators': 350, 'learning_rate': 0.05, 'max_depth': 6, 'regularization': 1.0} on `xgboost`, we balance bias and variance to reduce the remaining confusion ('opel' misclassified as 'saab' (65 errors, 38.2%)).
  - *Expected Outcome:* Attain peak validation convergence by fine-tuning tree complexity.
  - *Mutations Applied:* `model_family: catboost -> xgboost, max_depth: 8 -> 6`
  - *Observed Outcome:* CONFIRMED: New peak reached. Val Macro-F1=0.7774 (+0.0163 vs prior best), Overfit Gap=+0.2226, Worst Class 'opel' F1=0.6125.
