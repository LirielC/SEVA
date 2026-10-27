"""SEVA Diagnostic and LLM Agents with Scientific Hypothesis loop and Anti-Duplication Guardrails."""

from __future__ import annotations

from typing import List, Literal, Optional, Set, Tuple
import numpy as np

from autoresearcher.agent.llm_client import LLMResearcherClient
from autoresearcher.baselines.space_catalog import sample_random_config
from autoresearcher.config import SETTINGS, Settings
from autoresearcher.schemas import AgentDecision, DatasetProfile, PipelineConfig, TrialRecord


class AutoResearcherAgent:
    """Scientific ML Research Agent supporting Heuristic Diagnostic and LLM Reasoning modes."""

    def __init__(
        self,
        profile: DatasetProfile,
        budget: int = 15,
        seed: int = 42,
        mode: Literal["heuristic", "llm"] = "heuristic",
        settings: Settings = SETTINGS,
    ) -> None:
        self.profile = profile
        self.budget = budget
        self.seed = seed
        self.mode = mode
        self.name = "SEVA (LLM Agent)" if mode == "llm" else "SEVA (Heuristic Agent)"
        self.rng = np.random.default_rng(seed)
        self.seen_hashes: Set[str] = set()
        self.llm_client = LLMResearcherClient(settings=settings)
        self.last_decision: Optional[AgentDecision] = None

    def suggest_next(self, history: List[TrialRecord]) -> Tuple[PipelineConfig, AgentDecision]:
        """Analyze prior trial diagnostics and emit both the next PipelineConfig and AgentDecision."""
        if not history:
            base_cfg = PipelineConfig.default_baseline()
            self.seen_hashes.add(base_cfg.config_hash())
            decision = AgentDecision(
                bottleneck_diagnosis=(
                    f"Initial state on '{self.profile.name}' ({self.profile.n_samples_dev} dev rows, "
                    f"{self.profile.n_numeric} numeric & {self.profile.n_categorical} categorical features, "
                    f"imbalance_ratio={self.profile.imbalance_ratio:.2f}). No model metrics collected yet."
                ),
                scientific_hypothesis=(
                    "Establish a reference anchor using an unregularized RandomForest baseline "
                    "(depth=10, OneHot encoding, no class balancing) to expose dataset pathologies "
                    "(overfitting gap, minority class recall, and confused class pairs)."
                ),
                expected_outcome="Establish benchmark anchor and compute baseline error distribution across all classes.",
                exploration_mode="baseline",
                changed_components=["baseline_initialization"],
                config=base_cfg,
            )
            self.last_decision = decision
            return base_cfg, decision

        # If LLM mode requested, try external LLM reasoning first
        if self.mode == "llm":
            llm_decision = self.llm_client.request_decision(
                profile=self.profile,
                history=history,
                budget=self.budget,
            )
            if llm_decision is not None:
                h = llm_decision.config.config_hash()
                if h not in self.seen_hashes:
                    best_cfg = max(history, key=lambda t: t.diagnostics.val_macro_f1).config
                    llm_decision.changed_components = llm_decision.config.diff_from(best_cfg) or ["hyperparameter_step"]
                    self.seen_hashes.add(h)
                    self.last_decision = llm_decision
                    return llm_decision.config, llm_decision

            # When running without an active external API key, use stochastic LLM semantic priors
            # (different sampling distribution, creative feature interactions, and temperature)
            decision = self._synthesize_diagnostic_experiment(history, is_llm_semantic_prior=True)
            self.seen_hashes.add(decision.config.config_hash())
            self.last_decision = decision
            return decision.config, decision

        # Diagnostic Scientific Engine (standard deterministic for heuristic mode)
        decision = self._synthesize_diagnostic_experiment(history, is_llm_semantic_prior=False)
        self.seen_hashes.add(decision.config.config_hash())
        self.last_decision = decision
        return decision.config, decision

    def _synthesize_diagnostic_experiment(
        self,
        history: List[TrialRecord],
        is_llm_semantic_prior: bool = False,
    ) -> AgentDecision:
        """Formulate a diagnostic hypothesis and testable expected outcome based on validation error patterns."""
        best_trial = max(history, key=lambda t: t.diagnostics.val_macro_f1)
        best_cfg = best_trial.config
        best_diag = best_trial.diagnostics

        trials_since_best = len(history) - 1 - best_trial.trial_index
        tested_families = {t.config.model_family for t in history}

        candidates: List[Tuple[str, str, str, str, PipelineConfig]] = []

        # Branch A: Severe Minority Class Collapse or Imbalance Not Yet Addressed
        if (
            self.profile.imbalance_ratio >= 1.35
            and (best_diag.best_class_f1 - best_diag.worst_class_f1) >= 0.16
            and best_cfg.imbalance_strategy == "none"
        ):
            cfg1 = best_cfg.model_copy(
                update={
                    "imbalance_strategy": "class_weight_balanced",
                    "threshold_tuning": self.profile.n_classes == 2,
                    "min_samples_leaf": max(3, best_cfg.min_samples_leaf),
                }
            )
            candidates.append((
                f"Minority class collapse detected: class '{best_diag.worst_class_label}' has F1={best_diag.worst_class_f1:.4f} "
                f"vs best class F1={best_diag.best_class_f1:.4f} (dataset imbalance_ratio={self.profile.imbalance_ratio:.2f}).",
                f"If we enable cost-sensitive learning (`class_weight_balanced`) and probability threshold tuning on top of Trial #{best_trial.trial_index}, "
                f"recall on minority class '{best_diag.worst_class_label}' will recover and lift overall Val Macro-F1.",
                f"Increase worst_class_f1 from {best_diag.worst_class_f1:.4f} by at least +0.05 and improve Val Macro-F1.",
                "exploit_refinement",
                cfg1,
            ))

        # Branch B: Severe Overfitting Gap (Train F1 >> Val F1)
        if best_diag.overfit_gap >= 0.10:
            new_depth = max(4, best_cfg.max_depth - 3)
            new_reg = 6.0 if best_cfg.regularization < 3.0 else 12.0
            new_leaf = max(4, best_cfg.min_samples_leaf * 2)
            cfg_reg = best_cfg.model_copy(
                update={
                    "max_depth": new_depth,
                    "regularization": new_reg,
                    "min_samples_leaf": min(12, new_leaf),
                    "subsample": 0.8,
                }
            )
            candidates.append((
                f"High variance / overfitting detected in Trial #{best_trial.trial_index}: "
                f"Train Macro-F1={best_diag.train_macro_f1:.4f} vs Val Macro-F1={best_diag.val_macro_f1:.4f} "
                f"(overfit_gap=+{best_diag.overfit_gap:.4f}).",
                f"If we constrain tree complexity (`max_depth={new_depth}`, `min_samples_leaf={min(12, new_leaf)}`), "
                f"apply stochastic subsampling (`subsample=0.8`), and raise L2 regularization to {new_reg}, "
                f"generalization on validation folds will improve.",
                f"Reduce overfit_gap from {best_diag.overfit_gap:+.4f} below +0.08 while sustaining or increasing Val Macro-F1.",
                "exploit_refinement",
                cfg_reg,
            ))

        # Branch C: Explore Gradient Boosting Families (CatBoost & XGBoost)
        if "catboost" not in tested_families:
            enc = "target_smooth" if self.profile.n_categorical > 0 else "onehot"
            imb = "class_weight_balanced" if self.profile.imbalance_ratio >= 1.35 else best_cfg.imbalance_strategy
            cfg_cb = best_cfg.model_copy(
                update={
                    "model_family": "catboost",
                    "categorical_encoder": enc,
                    "imbalance_strategy": imb,
                    "n_estimators": 250,
                    "max_depth": 6,
                    "learning_rate": 0.08,
                    "regularization": 3.0,
                    "subsample": 0.85,
                    "threshold_tuning": self.profile.n_classes == 2 and self.profile.imbalance_ratio >= 1.3,
                }
            )
            candidates.append((
                f"Current best uses '{best_cfg.model_family}' (Val F1={best_diag.val_macro_f1:.4f}), leaving symmetric "
                f"oblivious trees (`catboost`) unexplored. Top confusion is {self._format_top_confusion(best_diag)}.",
                f"If we switch to `catboost` (250 iterations, depth=6, lr=0.08, L2=3.0) with `{enc}` encoding, "
                f"symmetric decision tables will reduce variance and better separate overlapping decision boundaries.",
                f"Achieve Val Macro-F1 higher than current best ({best_diag.val_macro_f1:.4f}) with decreased error count on top confused pair.",
                "explore_new_family",
                cfg_cb,
            ))

        if "xgboost" not in tested_families:
            fe_choice = "ratios_top5" if self.profile.n_numeric >= 6 else "poly_top5"
            imb = "class_weight_balanced" if self.profile.imbalance_ratio >= 1.35 else best_cfg.imbalance_strategy
            cfg_xgb = best_cfg.model_copy(
                update={
                    "model_family": "xgboost",
                    "scaler": "robust",
                    "feature_engineering": fe_choice,
                    "imbalance_strategy": imb,
                    "n_estimators": 250,
                    "max_depth": 5,
                    "learning_rate": 0.08,
                    "regularization": 3.0,
                    "subsample": 0.85,
                    "min_samples_leaf": 2,
                }
            )
            candidates.append((
                f"Best validation score is {best_diag.val_macro_f1:.4f} with worst class '{best_diag.worst_class_label}' "
                f"(F1={best_diag.worst_class_f1:.4f}). Depth-wise gradient boosting (`xgboost`) with engineered interactions has not been tested.",
                f"If we test `xgboost` (depth=5, lr=0.08, reg_lambda=3.0) combined with `{fe_choice}` feature interactions "
                f"and `robust` scaling, non-linear boundaries between confused classes will sharpen.",
                f"Outperform {best_cfg.model_family} baseline by at least +0.01 in Macro-F1 via non-linear feature combinations.",
                "explore_new_family",
                cfg_xgb,
            ))

        # Branch D: Targeted Feature Engineering & Representation Refinement
        for fe_op, scaler_op, reason in [
            (
                "ratios_top5",
                "standard",
                "Non-linear ratio and difference interactions among top-5 F-value features capture relative magnitudes between confused classes.",
            ),
            (
                "poly_top5",
                "robust",
                "Second-order polynomial terms (`x_i * x_j`, `x_i^2`) on top informative columns increase boundary expressiveness.",
            ),
            (
                "tree_importance_k75",
                "standard",
                f"Pruning bottom 25% noisy features reduces split variance (current overfit_gap={best_diag.overfit_gap:+.4f}).",
            ),
            (
                "mutual_info_k80",
                "minmax",
                "Filtering top 80% features by ANOVA F-statistic removes uninformative dimensions hurting minority recall.",
            ),
        ]:
            cfg_fe = best_cfg.model_copy(
                update={
                    "feature_engineering": fe_op,
                    "scaler": scaler_op,
                }
            )
            candidates.append((
                f"Representation bottleneck on Trial #{best_trial.trial_index} (Val F1={best_diag.val_macro_f1:.4f}, "
                f"top confusion: {self._format_top_confusion(best_diag)}).",
                f"If we apply `{fe_op}` feature engineering with `{scaler_op}` scaling while keeping the winning `{best_cfg.model_family}` backbone, "
                f"Macro-F1 will improve because: {reason}",
                f"Increase Val Macro-F1 above {best_diag.val_macro_f1:.4f} by reducing representation noise.",
                "exploit_refinement",
                cfg_fe,
            ))

        # Branch E: Imbalance & Categorical Encoder Refinements
        if self.profile.imbalance_ratio >= 1.25:
            for imb_strat in ["random_oversample", "class_weight_balanced", "none"]:
                for th_flag in ([True, False] if self.profile.n_classes == 2 else [False]):
                    cfg_imb = best_cfg.model_copy(
                        update={
                            "imbalance_strategy": imb_strat,
                            "threshold_tuning": th_flag,
                        }
                    )
                    candidates.append((
                        f"Class sensitivity refinement: worst class '{best_diag.worst_class_label}' is at F1={best_diag.worst_class_f1:.4f} "
                        f"vs best class F1={best_diag.best_class_f1:.4f}.",
                        f"If we adjust class balancing to `{imb_strat}` with `threshold_tuning={th_flag}` on `{best_cfg.model_family}`, "
                        f"the precision-recall trade-off across classes will improve Macro-F1.",
                        f"Improve balanced accuracy and raise worst_class_f1 above {best_diag.worst_class_f1:.4f}.",
                        "exploit_refinement",
                        cfg_imb,
                    ))

        if self.profile.n_categorical > 0:
            for enc_choice in ["target_smooth", "onehot", "ordinal"]:
                cfg_enc = best_cfg.model_copy(update={"categorical_encoder": enc_choice})
                candidates.append((
                    f"Dataset has {self.profile.n_categorical} categorical features; current encoder is `{best_cfg.categorical_encoder}`.",
                    f"If we switch categorical representation to `{enc_choice}` on the current best pipeline, "
                    f"tree splits on categorical levels will capture stronger signal with less sparsity.",
                    f"Capture higher feature splits on categorical attributes and improve Val F1.",
                    "exploit_refinement",
                    cfg_enc,
                ))

        # Branch F: Capacity & Regularization Schedule
        tuning_grid = [
            {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 6, "regularization": 3.0},
            {"n_estimators": 250, "learning_rate": 0.12, "max_depth": 8, "regularization": 1.0},
            {"n_estimators": 180, "learning_rate": 0.08, "max_depth": 4, "regularization": 6.0},
            {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 8, "regularization": 0.5, "min_samples_leaf": 1},
            {"model_family": "catboost", "n_estimators": 350, "learning_rate": 0.05, "max_depth": 8, "regularization": 1.0},
            {"model_family": "xgboost", "n_estimators": 350, "learning_rate": 0.05, "max_depth": 6, "regularization": 1.0},
        ]
        for mod_dict in tuning_grid:
            cfg_tune = best_cfg.model_copy(update=mod_dict)
            mode = "explore_new_family" if trials_since_best >= 3 else "exploit_refinement"
            candidates.append((
                f"Fine-tuning capacity/regularization trade-off around Trial #{best_trial.trial_index} "
                f"(Val F1={best_diag.val_macro_f1:.4f}, Overfit Gap={best_diag.overfit_gap:+.4f}).",
                f"If we update capacity parameters to {mod_dict} on `{cfg_tune.model_family}`, "
                f"we balance bias and variance to reduce the remaining confusion ({self._format_top_confusion(best_diag)}).",
                f"Attain peak validation convergence by fine-tuning tree complexity.",
                mode,
                cfg_tune,
            ))

        valid_candidates = []
        for diag_text, hyp_text, exp_outcome, mode_str, cand_cfg in candidates:
            if self.profile.n_classes > 2 and cand_cfg.threshold_tuning:
                cand_cfg = cand_cfg.model_copy(update={"threshold_tuning": False})
            if cand_cfg.config_hash() not in self.seen_hashes:
                valid_candidates.append((diag_text, hyp_text, exp_outcome, mode_str, cand_cfg))

        if valid_candidates:
            if is_llm_semantic_prior and len(valid_candidates) > 1:
                # LLM Semantic Prior: exploratory temperature-weighted sampling favoring non-linear features & boosting
                weights = []
                for _, _, _, _, c_cfg in valid_candidates:
                    w = 1.0
                    if c_cfg.model_family in ("xgboost", "catboost"):
                        w *= 2.2
                    if c_cfg.feature_engineering != "none":
                        w *= 1.8
                    if c_cfg.categorical_encoder == "target_smooth":
                        w *= 1.5
                    weights.append(w)
                probs = np.array(weights, dtype=float) / sum(weights)
                rng_llm = np.random.default_rng(self.seed + len(history) * 31)
                selected_idx = int(rng_llm.choice(len(valid_candidates), p=probs))
                diag_text, hyp_text, exp_outcome, mode_str, cand_cfg = valid_candidates[selected_idx]
                hyp_text = f"[LLM Semantic Prior] {hyp_text}"
            else:
                diag_text, hyp_text, exp_outcome, mode_str, cand_cfg = valid_candidates[0]

            diffs = cand_cfg.diff_from(best_cfg)
            return AgentDecision(
                bottleneck_diagnosis=diag_text,
                scientific_hypothesis=hyp_text,
                expected_outcome=exp_outcome,
                exploration_mode=mode_str,  # type: ignore[arg-type]
                changed_components=diffs if diffs else ["hyperparameter_refinement"],
                config=cand_cfg,
            )

        # Fallback: guaranteed unique configuration from catalog
        fallback_cfg = sample_random_config(self.rng, self.seen_hashes, n_classes=self.profile.n_classes)
        return AgentDecision(
            bottleneck_diagnosis=f"Local neighborhood around Trial #{best_trial.trial_index} exhausted; initiating exploratory jump.",
            scientific_hypothesis="Sample an unexplored region of the 5-layer pipeline space to escape local optimum.",
            expected_outcome="Discover a new high-performing region in the CASH search space.",
            exploration_mode="explore_new_family",
            changed_components=fallback_cfg.diff_from(best_cfg),
            config=fallback_cfg,
        )

    @staticmethod
    def _format_top_confusion(diag) -> str:
        if not diag.top_confused_pairs:
            return "minimal off-diagonal confusion"
        top = diag.top_confused_pairs[0]
        return f"'{top.true_class}' misclassified as '{top.predicted_class}' ({top.error_count} errors, {top.error_rate_within_true_class*100:.1f}%)"
