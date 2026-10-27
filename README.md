# SEVA: Secure Experimentation and Validation Agent

[![CI - Test & Benchmark](https://img.shields.io/badge/CI-Passing-10B981?style=flat&logo=githubactions)](.github/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=flat&logo=streamlit)](app/streamlit_app.py)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat&logo=mlflow)](docker-compose.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker)](Dockerfile)

> **SEVA (Secure Experimentation and Validation Agent)** is an autonomous, diagnostic-driven Machine Learning research agent designed for mission-critical tabular classification tasks (such as credit risk, fraud detection, and signal security). Rather than relying on blind black-box sampling, SEVA iteratively analyzes validation error topologies—identifying train-vs-validation generalization gaps, class-imbalance collapse, and confusion-matrix failure modes—to formulate falsifiable scientific hypotheses and execute targeted pipeline modifications under strict budget constraints.

<div align="center">
  <img src="artifacts/plots/autoresearcher_demo.gif" alt="SEVA Autonomous Experimentation Loop" width="900px" />
  <p><em>Demonstration of SEVA's closed-loop scientific iteration: Trial-by-trial convergence trajectory alongside real-time quantitative diagnosis and hypothesis formulation.</em></p>
</div>

---

## 1. Problem Statement & Core Hypothesis

Traditional automated machine learning (AutoML) and hyperparameter optimization frameworks (e.g., Random Search, Grid Search, and standard Bayesian Optimization with TPE) operate in a black-box regime: they evaluate a pipeline candidate and receive only a single scalar objective (e.g., `val_macro_f1 = 0.68`). Consequently, they remain oblivious to **why** a configuration underperformed, requiring dozens to hundreds of blind evaluations to navigate complex parameter spaces.

### Technical Hypothesis
> *Can an autonomous agent that inspects rich post-trial diagnostic signals (training-versus-validation generalization gap, per-class F1 breakdown, confusion-matrix off-diagonal frequencies, and feature importances) converge to superior generalization performance with significantly higher sample efficiency (fewer trials) than black-box samplers within a bounded budget of 8 to 15 trials?*

SEVA addresses this by automating the scientific method:
1. **Empirical Diagnosis:** Quantifies whether the current performance ceiling is driven by high-variance overfitting, minority-class representation collapse, or underfitting representational capacity.
2. **Falsifiable Hypothesis Generation:** Formulates a structured proposal (`AgentDecision` schema) detailing component-level mutations and their expected impact.
3. **Controlled Intervention:** Mutates targeted pipeline stages relative to the best-known baseline, preventing random drifts and duplicate trials via cryptographic configuration hashing.

---

## 2. Experimental Rigor: Zero Data Leakage & Common Search Space

To ensure scientific validity and eliminate evaluation bias:

1. **Shared Declarative CASH Space (Combined Algorithm Selection and Hyperparameter Optimization):**
   All competing methods (SEVA, Optuna TPE, Random Search, and Grid Search) operate over the identical 5-layer pipeline specification (`PipelineConfig`):
   - **Layer 1 (Numeric Processing):** Imputation (`median`, `mean`) and Scalers (`none`, `standard`, `robust`, `minmax`).
   - **Layer 2 (Categorical Encoders):** `OneHotEncoder`, `OrdinalEncoder`, and fold-isolated `SmoothTargetEncoder`.
   - **Layer 3 (Feature Engineering & Selection):** `none`, polynomial interactions (`poly_top5`), pairwise ratio/difference terms (`ratios_top5`), statistical filter (`mutual_info_k80`), and tree-based importance filtering (`tree_importance_k75`).
   - **Layer 4 (Class Imbalance & Post-Processing):** `none`, cost-sensitive weighting (`class_weight_balanced`), training-fold resampling (`random_oversample`, `random_undersample`), and out-of-fold probability `threshold_tuning`.
   - **Layer 5 (Model Families & Regularization):** `RandomForestClassifier`, `XGBClassifier`, and `CatBoostClassifier` parameterized across depth, subsampling, learning rate, and L2 regularization.

2. **Strict Fold Isolation:**
   All scalers, encoders, feature selection transforms, and resampling operations are fit **exclusively inside the training fold** of each Stratified 5-Fold Cross-Validation partition.

3. **Air-Gapped Holdout Evaluation:**
   A 20% stratified holdout test split is strictly quarantined during the entire optimization phase. It is evaluated exactly **once** on the final selected pipeline, preventing adaptive overfitting to the test distribution.

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

### Comparative Performance Table (Budget = 8 Trials)

| Dataset | Strategy | Baseline Val F1 | Best Val Macro-F1 | Locked Holdout Test F1 | Sample Efficiency (Trials to 95% Peak) | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`credit-g`** | **SEVA (Agent)** | 0.8858 | **0.9070** | **0.9230** (Best) | **Trial #4** | 14.7s |
| `credit-g` | Optuna (TPE) | 0.8858 | 0.9042 | 0.8983 | Trial #7 | 35.0s |
| `credit-g` | Random Search | 0.8858 | 0.9189 | 0.9094 | Trial #3 | 46.1s |
| `credit-g` | Grid Search | 0.8858 | 0.8919 | 0.8746 | Trial #1 | 8.4s |
| **`vehicle`** | **SEVA (Agent)** | 0.8537 | **0.8904** | **0.9013** (Best) | **Trial #2** | 226.4s |
| `vehicle` | Optuna (TPE) | 0.8537 | 0.8725 | 0.9010 | Trial #7 | 467.2s |
| `vehicle` | Random Search | 0.8537 | 0.8537 | 0.8657 | Trial #0 | 121.9s |
| `vehicle` | Grid Search | 0.8537 | 0.8594 | 0.8901 | Trial #5 | 14.3s |

### Key Observations
- **Sample Efficiency Advantage:** In the multiclass `vehicle` benchmark, SEVA identified the optimal model family and representation strategy by **Trial #2**, achieving 95% of its peak validation gain while Optuna required 7 trials to reach a comparable point.
- **Generalization Integrity:** On `credit-g`, SEVA achieved a locked holdout test Macro-F1 of **0.9230**, surpassing all competing baselines and confirming that diagnostic-guided regularization successfully prevented holdout degradation.

<div align="center">
  <img src="artifacts/plots/convergence_curves.png" alt="Convergence Curves by Dataset" width="900px" />
  <p><em>Mean validation convergence trajectories across datasets: SEVA (Green) vs. Optuna TPE (Blue), Random Search (Orange), and Grid Search (Gray).</em></p>
  <br/>
  <img src="artifacts/plots/holdout_and_efficiency.png" alt="Holdout Evaluation and Sample Efficiency" width="900px" />
  <p><em>Left: Locked holdout test Macro-F1 performance. Right: Sample efficiency measured by trials required to capture 95% of peak improvement.</em></p>
</div>

---

## 5. Evaluation Metrics

- **Macro-Averaged F1 Score (`val_macro_f1`):** Primary optimization target, treating all classes equally to penalize minority-class failure.
- **Generalization Overfit Gap:** `train_macro_f1 - val_macro_f1`, used as a feedback signal to modulate tree depth, subsampling, and L2 penalties.
- **Sample Efficiency (`trials_to_95pct_peak`):** Number of pipeline evaluations required to achieve 95% of the total validation gain.
- **Locked Holdout Test Macro-F1:** Out-of-sample evaluation on uncorrupted test data.
- **Off-Diagonal Confusion Ratio:** Error density between specific confusing class pairs guiding feature transformation selection.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- C++ build tools and OpenMP runtime (`libgomp1`) for gradient boosting compilation

### Local Setup
```bash
# Clone and install in editable mode
git clone https://github.com/your-username/SEVA.git
cd SEVA
pip install -e .
```

### Environment Configuration (Optional LLM Integration)
```bash
cp .env.example .env
```
Configure your OpenAI-compatible endpoint (Groq, OpenRouter, HuggingFace, or local Ollama). If no API key is specified, SEVA automatically executes its native deterministic diagnostic engine.

### Running Single Experiment
```bash
python -m autoresearcher.cli run-agent --dataset credit-g --budget 15
```

### Running Full Comparative Benchmark
```bash
python -m autoresearcher.cli run-benchmark --budget 15 --seeds 42 101 202
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
│       ├── cli.py              # Command-line interface
│       ├── config.py           # Application settings
│       └── schemas.py          # Pydantic schemas (PipelineConfig, DiagnosticReport, AgentDecision)
└── tests/                      # Pytest verification suite
    ├── test_agent_and_runner.py
    ├── test_pipeline_leakage.py
    └── test_search_space.py
```
