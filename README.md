# SEVA: Secure Experimentation and Validation Agent

[![CI - Test & Benchmark](https://img.shields.io/badge/CI-Passing-10B981?style=flat&logo=githubactions)](.github/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space%20Demo-yellow)](https://huggingface.co/spaces/Archiel89/SEVA)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=flat&logo=streamlit)](app/streamlit_app.py)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat&logo=mlflow)](docker-compose.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker)](Dockerfile)

> **SEVA (Secure Experimentation and Validation Agent)** is an autonomous, diagnostic-driven Machine Learning research agent designed for mission-critical tabular classification tasks (such as credit risk, fraud detection, and signal security). Rather than relying on blind black-box sampling, SEVA iteratively analyzes validation error topologies—identifying train-vs-validation generalization gaps, class-imbalance collapse, and confusion-matrix failure modes—to formulate falsifiable scientific hypotheses and execute targeted pipeline modifications under strict budget constraints.

<div align="center">
  <img src="artifacts/plots/autoresearcher_demo.gif" alt="SEVA Autonomous Experimentation Loop" width="900px" />
  <p><em>Demonstration of SEVA's closed-loop scientific iteration: Trial-by-trial convergence trajectory alongside real-time quantitative diagnosis, hypothesis formulation, and post-evaluation observed outcome verification.</em></p>
</div>

---

## 1. Problem Statement & Core Hypothesis

Traditional automated machine learning (AutoML) and hyperparameter optimization frameworks (e.g., Random Search, Grid Search, and standard Bayesian Optimization with Optuna TPE) operate in a black-box regime: they evaluate a pipeline candidate and receive only a single scalar objective (e.g., `val_macro_f1 = 0.68`). Consequently, they remain oblivious to **why** a configuration underperformed, requiring dozens to hundreds of blind evaluations to navigate complex parameter spaces.

### Technical Hypothesis
> *Can an autonomous agent that inspects rich post-trial diagnostic signals (training-versus-validation generalization gap, per-class F1 breakdown, confusion-matrix off-diagonal frequencies, and feature importances) converge to superior generalization performance with significantly higher sample efficiency (fewer trials) than black-box samplers within a bounded budget of 6 to 15 trials?*

SEVA addresses this by automating the scientific method:
1. **Empirical Diagnosis:** Quantifies whether the current performance ceiling is driven by high-variance overfitting, minority-class representation collapse, or underfitting representational capacity.
2. **Falsifiable Hypothesis Generation:** Formulates a structured proposal (`AgentDecision` schema) detailing component-level mutations and their expected impact.
3. **Controlled Intervention:** Mutates targeted pipeline stages relative to the best-known baseline, preventing random drifts and duplicate trials via cryptographic configuration hashing.
4. **Observed Outcome Verification:** Compares post-evaluation cross-validation metrics against the pre-trial expectation, recording whether the scientific hypothesis was confirmed or refuted.

---

## 2. Experimental Rigor & Four-Strategy Ablation

To isolate the benefit of diagnostic-driven optimization from the specific contribution of Large Language Model reasoning, SEVA natively benchmarks four competing strategies under an identical evaluation budget and identical stratified splits:

1. **Random Search:** Uniform random sampling across the declarative search space.
2. **Optuna (TPE):** Standard black-box Bayesian Optimization via multivariate Tree-structured Parzen Estimator.
3. **SEVA (Heuristic Agent):** Deterministic diagnostic expert engine that diagnoses overfitting gaps, class collapse, and confusion pairs to apply targeted pipeline mutations.
4. **SEVA (LLM Agent):** Reasoning agent that prompts an instruction-tuned LLM (e.g., Qwen2.5-7B-Instruct) with complete lab diagnostics, using Pydantic schema validation and automatic self-repair.

### Zero Data Leakage Architecture
- **Shared Declarative CASH Space:** All four strategies explore the identical 5-layer pipeline specification (`PipelineConfig`):
  - **Layer 1 (Numeric Processing):** Imputation (`median`, `mean`) and Scalers (`none`, `standard`, `robust`, `minmax`).
  - **Layer 2 (Categorical Encoders):** `OneHotEncoder`, `OrdinalEncoder`, and fold-isolated `SmoothTargetEncoder`.
  - **Layer 3 (Feature Engineering & Selection):** `none`, polynomial interactions (`poly_top5`), pairwise ratio/difference terms (`ratios_top5`), statistical filter (`mutual_info_k80`), and tree-based importance filtering (`tree_importance_k75`).
  - **Layer 4 (Class Imbalance & Post-Processing):** `none`, cost-sensitive weighting (`class_weight_balanced`), training-fold resampling (`random_oversample`, `random_undersample`), and out-of-fold probability `threshold_tuning`.
  - **Layer 5 (Model Families & Regularization):** `RandomForestClassifier`, `XGBClassifier`, and `CatBoostClassifier` parameterized across depth, subsampling, learning rate, and L2 regularization.
- **Strict Fold Isolation:** All transformers, encoders, and resampling routines are fitted **exclusively inside the training fold** of each Stratified 5-Fold Cross-Validation split.
- **Air-Gapped Holdout Evaluation:** A 20% stratified holdout test split is strictly quarantined during the entire optimization phase and evaluated **once** on the final selected pipeline.

---

## 3. Technology Stack

- **Core Framework & Models:** Python 3.12, Scikit-learn (1.9+), XGBoost (3.4+), CatBoost (1.2+).
- **Optimization Baselines:** Optuna (5.0+) with multivariate `TPESampler`, uniform `RandomSearch`, and deterministic `GridSearch`.
- **Reasoning Engine:** OpenAI-compatible client interface supporting Qwen2.5-7B-Instruct (via OpenRouter, Groq, HuggingFace Serverless, or local Ollama) paired with an embedded deterministic diagnostic reasoning engine for offline/CI execution.
- **Validation & Schemas:** Pydantic (v2) for typed data integrity, runtime validation, and deterministic SHA-256 configuration hashing.
- **Experiment Tracking:** MLflow (tracking server and artifact store) alongside local JSONL Lab Notebook audit logs.
- **Visualization & UI:** Streamlit for interactive telemetry, Matplotlib/Seaborn for publication-grade convergence and sample efficiency charts.
- **DevOps & Delivery:** Docker, Docker Compose, Pytest, GitHub Actions CI.

---

## 4. Benchmark Results & Quantitative Evaluation

The system was evaluated across three OpenML tabular datasets with distinct structural challenges:
- **`credit-g` (OpenML ID 31):** Mixed categorical/numeric features, severe class imbalance (70/30), high propensity for tree overfitting.
- **`phoneme` (OpenML ID 1489):** Non-linear continuous feature interactions with acoustic boundary overlap.
- **`vehicle` (OpenML ID 54):** 4-class multiclass silhouette classification with high inter-class confusion between specific pairs (e.g., `opel` vs. `saab`).

### Four-Strategy Ablation Benchmark Table (Budget = 6 Trials on Real OpenML Datasets)

| Dataset | Strategy | Baseline Val F1 | Best Val Macro-F1 | Locked Holdout Test F1 | Sample Efficiency (Trials to 95% Peak) | AUC Convergence | LLM Cost ($) | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`credit-g`** | **SEVA (Heuristic Agent)** | 0.6572 | **0.7205** (Best) | **0.6790** | **Trial #2** | **0.7075** | N/A | 7.9s |
| **`credit-g`** | **SEVA (LLM Agent)** | 0.6572 | **0.7161** | **0.6858** (Best) | **Trial #3** | 0.6867 | $0.0000 | 8.0s |
| `credit-g` | Optuna (TPE) | 0.6572 | 0.6943 | 0.6755 | Trial #4 | 0.6740 | N/A | 12.9s |
| `credit-g` | Random Search | 0.6572 | 0.7007 | 0.6681 | Trial #4 | 0.6829 | N/A | 14.0s |
| **`phoneme`** | **SEVA (Heuristic Agent)** | 0.8345 | **0.8520** (Best) | **0.8677** | Trial #5 | 0.8381 | N/A | 27.6s |
| **`phoneme`** | **SEVA (LLM Agent)** | 0.8345 | **0.8476** | **0.8812** (Best) | **Trial #2** | **0.8432** | $0.0000 | 70.0s |
| `phoneme` | Optuna (TPE) | 0.8345 | 0.8466 | 0.8376 | Trial #5 | 0.8365 | N/A | 27.5s |
| `phoneme` | Random Search | 0.8345 | 0.8434 | 0.8740 | Trial #5 | 0.8362 | N/A | 32.1s |
| **`vehicle`** | **SEVA (Heuristic Agent)** | 0.7368 | **0.7842** (Best) | **0.7893** (Best) | Trial #5 | 0.7565 | N/A | 22.7s |
| **`vehicle`** | **SEVA (LLM Agent)** | 0.7368 | **0.7774** | 0.7432 | Trial #5 | 0.7570 | $0.0000 | 59.2s |
| `vehicle` | Optuna (TPE) | 0.7368 | 0.7734 | 0.7437 | Trial #1 | 0.7673 | N/A | 266.1s |
| `vehicle` | Random Search | 0.7368 | 0.7625 | 0.7766 | Trial #5 | 0.7481 | N/A | 62.9s |

### Key Observations & Scientific Scope
- **Distinct Trajectories:** SEVA (Heuristic Agent) and SEVA (LLM Agent) exhibit distinct exploration pathways. On `credit-g`, the Heuristic Agent reached peak validation Macro-F1 of **0.7205** at Trial #2 via cost-sensitive balancing, while the LLM Agent achieved the highest Holdout generalization at **0.6858**.
- **Superiority over Black-Box Solvers:** On `credit-g`, Optuna and Random Search stagnated at 0.6943 and 0.7007 respectively, falling short of both SEVA agents. On `phoneme`, the LLM agent reached **0.8812** on the locked holdout test.
- **Strict Holdout Air-Gap:** All preprocessing, feature engineering, and class rebalancing transforms are strictly fitted within cross-validation training folds. The holdout test scores faithfully reflect real-world generalization on unseen distributions.
- **Experimental Scope & Scientific Disclaimer:** The benchmark metrics presented above reflect a single controlled reference seed (`seed=42`) across three OpenML benchmark datasets with a fixed budget of 6 trials. With a single seed and three datasets, these findings constitute **promising preliminary empirical evidence** of diagnostic search efficiency rather than a definitive demonstration of universally generalizable superiority over Bayesian optimization. SEVA includes built-in support for multi-seed evaluations (`--seeds 42 101 202`) and deterministic replay (`seva replay`) to enable multi-seed statistical significance testing (e.g., Wilcoxon signed-rank tests).

<div align="center">
  <img src="artifacts/plots/convergence_curves.png" alt="Convergence Curves by Dataset" width="900px" />
  <p><em>Mean validation convergence trajectories across datasets: SEVA (Green/Purple) vs. Optuna TPE (Blue), Random Search (Orange), and Grid Search (Gray).</em></p>
  <br/>
  <img src="artifacts/plots/holdout_and_efficiency.png" alt="Holdout Evaluation and Sample Efficiency" width="900px" />
  <p><em>Left: Locked holdout test Macro-F1 performance. Right: Sample efficiency measured by trials required to capture 95% of peak improvement.</em></p>
</div>

---

## 5. Evaluation Methodology, Metrics & Rigorous Protocols

### 5.1 Scientific Scope & Generalizability Disclaimer
With an evaluation budget of only six trials ($T=6$) per strategy across three datasets on a single reference seed (`seed=42`), the empirical findings presented in Section 4 represent **promising preliminary evidence** of diagnostic search efficiency rather than a definitive demonstration of universally generalizable superiority. A 6-trial budget illustrates early sample efficiency, but broader benchmarks with higher budgets (e.g., 20–50 trials) and multiple random seeds are essential for comprehensive statistical validation.

### 5.2 Definition of Search Trials
Each optimization strategy executes under an identical evaluation budget of 6 trials ($t \in \{0, 1, 2, 3, 4, 5\}$):
- **Trial #0 (Deterministic Baseline Anchor):** Identical across all four strategies. Evaluates an unregularized `RandomForestClassifier` (depth=10, `OneHotEncoder`, no class balancing, no resampling, no polynomial features) to expose baseline error topologies (train-val generalization gap, minority class recall, and confused class pairs).
- **Trials #1 to #5 (Strategy-Specific Search Iterations):**
  - **Random Search:** Samples 5 subsequent configurations independently and uniformly from the declarative 5-layer CASH space.
  - **Optuna (TPE):** Uses Tree-structured Parzen Estimators (`TPESampler(multivariate=True, seed=42)`) to construct probabilistic surrogate models $p(x|y)$ over prior trial validation scores.
  - **SEVA (Heuristic Agent):** Inspects the previous trial diagnostic payload and applies deterministic sequential rule-based interventions to resolve diagnosed pathologies (e.g., severe class collapse $\rightarrow$ cost-sensitive weights and threshold tuning; generalization gap $\Delta_{\text{overfit}} > 0.15 \rightarrow$ tree depth reduction, L2 regularization, and feature subsampling; low capacity $\rightarrow$ non-linear gradient boosting).
  - **SEVA (LLM Agent):** Serializes the diagnostic payload into a structured scientific prompt for the LLM (`Qwen2.5-Instruct`), generating a structured JSON mutation proposal that is validated and self-repaired via Pydantic schemas.

### 5.3 Data Partitioning, Seeds & Cross-Validation Folds
- **Reference Seed:** `seed=42` is fixed across all data splitters, fold generators, and stochastic baseline samplers to ensure identical experimental conditions.
- **Stratified Partitioning:** Each OpenML dataset is partitioned upfront into:
  - **80% Development Partition:** Exclusively used for model exploration, preprocessor fitting, and hyperparameter optimization.
  - **20% Holdout Test Partition:** Quarantined immediately after download and completely locked during the entire search loop.
- **5-Fold Cross-Validation:** The 80% Development partition is split into 5 stratified folds (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). All pipeline stages (imputation, scaling, one-hot encoding, SMOTE/resampling, and feature selection) are fit **exclusively inside the 4 training folds** and applied out-of-fold to the 5th validation fold. No test or validation statistics ever leak into training.

### 5.4 Strict Holdout Air-Gap Preservation
The 20% holdout test partition is **strictly air-gapped from all development and search decisions**:
- The holdout partition is **never** seen by any imputer, scaler, feature selector, or classifier during the 6 search trials.
- The holdout partition is **never** used by the Heuristic Agent or the LLM Agent to diagnose errors, formulate hypotheses, or mutate pipeline configurations.
- The holdout partition is **never** used to select winning configurations or tune decision thresholds.
- **Single Holdout Evaluation:** Only after a strategy finishes its entire 6-trial search budget is the single winning pipeline retrained on the full 80% Development partition and evaluated **exactly once** on the untouched holdout test partition to verify true out-of-sample generalization.

### 5.5 Model Selection Protocol (Within and Across Strategies)
- **Within-Strategy Selection:** For any given strategy, the winning pipeline is chosen strictly by the highest arithmetic mean Macro-F1 across the 5 cross-validation folds on the Development partition:
  $$\text{val\_macro\_f1}_t = \frac{1}{5} \sum_{k=1}^5 \text{Macro-F1}_{k, t}, \quad \text{Winning Pipeline} = \arg\max_{t} (\text{val\_macro\_f1}_t)$$
- **Cross-Strategy Selection & Comparison:** In the benchmark report and tables, the cross-strategy ranking is determined by comparing their respective peak validation CV scores (`Best Val Macro-F1`). The locked holdout test score is reported purely as an uncorrupted audit metric of generalization, never as a selection criterion.

### 5.6 Calculation of Convergence & Sample Efficiency Metrics
- **AUC-Convergence (Normalized Discrete Integral):** Quantifies search trajectory efficiency and cumulative regret across the search budget $T=6$. It is calculated as the arithmetic mean of the cumulative best validation score across all budget steps:
  $$\text{AUC-Convergence} = \frac{1}{T} \sum_{t=0}^{T-1} y^*_t, \quad \text{where} \quad y^*_t = \max_{0 \le i \le t} (\text{val\_macro\_f1}_i)$$
  Because the discrete area under the step-function trajectory is divided by the budget $T$, the metric is normalized into $[0.0, 1.0]$ and directly comparable to F1 scores. An agent that discovers high-performing configurations early (e.g., Trial #1 or #2) achieves a higher normalized cumulative score than an optimizer that only discovers a strong model in Trial #5.
- **Sample Efficiency (`Trials to 95% Peak` vs. `Trials to Target Threshold`):**
  - **`Trials to 95% Peak`:** Earliest trial index $t$ that captured at least 95% of the total validation improvement achieved over the Trial 0 baseline:
    $$\text{Target} = \text{val\_macro\_f1}_0 + 0.95 \times \left( \max_{0 \le i < T} \text{val\_macro\_f1}_i - \text{val\_macro\_f1}_0 \right)$$
  - **`Trials to Target Threshold`:** Earliest trial index $t$ reaching a user-defined absolute threshold (`--target-threshold <float>`).
- **Generalization Overfit Gap:** $\Delta_{\text{overfit}} = \text{train\_macro\_f1} - \text{val\_macro\_f1}$, computed across folds and supplied to the agent to diagnose variance pathologies.
- **LLM Operational Telemetry:** Tracks input tokens, output tokens, total tokens, call latency, and estimated USD cost based on standard model pricing.

### 5.7 Execution Environment, Library Versions & Reproducibility Logs
- **Hardware & Software Environment:**
  - Python: `3.12.10`
  - Scikit-learn: `1.9.1`
  - XGBoost: `3.4.1`
  - CatBoost: `1.2.10`
  - Optuna: `5.0.0`
  - Imbalanced-learn: `0.14.1`
  - Pydantic: `2.14.0`
- **Per-Trial JSON Logs:** Every trial execution is deterministically logged in `artifacts/runs/<dataset>__<strategy>__seed<seed>.json` with full hyperparameters, fold-level scores, confusion matrices, agent hypotheses, and empirical outcomes.
- **Deterministic Replay:** Reproduce any pipeline run identically using the CLI:
  ```bash
  seva replay --config artifacts/runs/credit-g__seva_heuristic_agent__seed42.json --dataset credit-g --seed 42
  ```

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- C++ build tools and OpenMP runtime (`libgomp1`) for gradient boosting compilation

### Local Setup
```bash
# Clone and install in editable mode
git clone https://github.com/LirielC/SEVA.git
cd SEVA
pip install -e .
```

### Environment Configuration (Optional LLM Integration)
```bash
cp .env.example .env
```
Configure your OpenAI-compatible endpoint (Groq, OpenRouter, HuggingFace, or local Ollama). If no API key is specified, SEVA automatically executes its native deterministic diagnostic engine.

### Running Single Experiment (Heuristic or LLM Mode)
```bash
# Run SEVA Heuristic Diagnostic Agent
seva run-agent --dataset credit-g --budget 12 --mode heuristic

# Run SEVA with external LLM reasoning
seva run-agent --dataset credit-g --budget 12 --mode llm
```

### Running 4-Way Comparative Benchmark
```bash
seva run-benchmark --budget 12 --seeds 42 101 202 --include-llm
```

### Deterministic Replay (Reproducibility Verification)
Faithfully reproduce and evaluate any winning pipeline configuration from its serialized JSON file or parameter dictionary:
```bash
seva replay --config artifacts/runs/credit-g__seva_heuristic_agent__seed42.json --dataset credit-g --seed 42
```

### Launching Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Running Automated Test Suite
```bash
pytest -v
```

---

## 7. Containerized Deployment (Docker Compose)

Deploy both the Streamlit Dashboard and the MLflow Tracking Server with a single command:

```bash
docker compose up --build
```

- **Streamlit Analytics Dashboard:** [http://localhost:8501](http://localhost:8501)
- **MLflow Tracking Server:** [http://localhost:5000](http://localhost:5000)

---

## 8. Repository Structure

```text
SEVA/
├── .github/workflows/ci.yml    # GitHub Actions CI workflow (pytest + benchmark smoke test)
├── app/
│   └── streamlit_app.py        # Interactive Streamlit analytics dashboard
├── artifacts/
│   ├── BENCHMARK_REPORT.md     # Auto-generated scientific markdown report
│   ├── plots/                  # Generated plots and animated demo GIF
│   └── runs/                   # Serialized JSON audit logs for each evaluated run
├── docker-compose.yml          # Multi-container orchestration (Streamlit + MLflow)
├── Dockerfile                  # Production container definition
├── LICENSE                     # MIT License specification
├── pyproject.toml              # Build backend and project metadata
├── requirements.txt            # Pinned dependencies
├── scripts/
│   └── generate_demo_gif.py    # Script for building the trial-by-trial animated GIF
├── src/
│   └── autoresearcher/
│       ├── agent/              # SEVA reasoning agent, prompt templates, and LLM client
│       ├── baselines/          # Optuna (TPE), Random Search, and Grid Search implementations
│       ├── benchmark/          # Multi-seed benchmark execution and markdown report generator
│       ├── data/               # OpenML dataset fetching, profiling, and leak-free splitters
│       ├── pipeline/           # 5-layer pipeline builder, fold transformers, and CV evaluator
│       ├── tracking/           # MLflow logging and Matplotlib visualization utilities
│       ├── cli.py              # Command-line interface (seva)
│       ├── config.py           # Application settings
│       └── schemas.py          # Pydantic schemas (PipelineConfig, DiagnosticReport, AgentDecision)
└── tests/                      # Pytest verification suite
    ├── test_agent_and_runner.py
    ├── test_pipeline_leakage.py
    └── test_search_space.py
```

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
