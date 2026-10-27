"""Generate an animated GIF showing SEVA's trial-by-trial scientific progress with 4-way comparison."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "artifacts" / "runs"
PLOTS_DIR = ROOT_DIR / "artifacts" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_animated_gif() -> Path:
    # Prefer new SEVA agent files, fallback to older naming if needed
    agent_file = (
        next(RUNS_DIR.glob("credit-g__seva_heuristic_agent__seed*.json"), None)
        or next(RUNS_DIR.glob("credit-g__seva_llm_agent__seed*.json"), None)
        or next(RUNS_DIR.glob("credit-g__autoresearcher_agent__seed*.json"), None)
    )
    if not agent_file:
        raise FileNotFoundError("Agent run file not found in artifacts/runs")

    agent_data = json.loads(agent_file.read_text(encoding="utf-8"))
    raw_trials = agent_data["trials"]

    optuna_file = next(RUNS_DIR.glob("credit-g__optuna_tpe__seed*.json"), None)
    random_file = next(RUNS_DIR.glob("credit-g__random_search__seed*.json"), None)

    optuna_curve = (
        [t["best_val_macro_f1_so_far"] for t in json.loads(optuna_file.read_text(encoding="utf-8"))["trials"]]
        if optuna_file
        else []
    )
    random_curve = (
        [t["best_val_macro_f1_so_far"] for t in json.loads(random_file.read_text(encoding="utf-8"))["trials"]]
        if random_file
        else []
    )

    n_trials = len(raw_trials)
    if optuna_curve:
        n_trials = min(n_trials, len(optuna_curve))
    if random_curve:
        n_trials = min(n_trials, len(random_curve))

    trials = raw_trials[:n_trials]
    agent_curve = [t["best_val_macro_f1_so_far"] for t in trials]
    optuna_curve = optuna_curve[:n_trials]
    random_curve = random_curve[:n_trials]

    frames: list[Image.Image] = []
    tmp_frame_dir = PLOTS_DIR / "_temp_frames"
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)

    for step in range(n_trials):
        current_t = trials[step]
        diag = current_t["diagnostics"]
        dec = current_t.get("agent_decision") or {}
        diag_text = dec.get("bottleneck_diagnosis", diag.get("diagnostic_summary_text", ""))
        hyp_text = dec.get("scientific_hypothesis", "Testing regularized pipeline configuration.")
        exp_text = dec.get("expected_outcome", "Improve validation Macro-F1.")
        obs_text = current_t.get("observed_outcome") or f"Val Macro-F1 = {diag['val_macro_f1']:.4f}"
        model_name = current_t["config"]["model_family"].upper()

        fig = plt.figure(figsize=(10.5, 6.2), dpi=120)
        fig.patch.set_facecolor("#0F172A")  # Deep Navy Slate background

        # Subplot 1: Convergence Curves
        ax1 = fig.add_axes([0.08, 0.44, 0.88, 0.46])
        ax1.set_facecolor("#1E293B")

        steps_range = np.arange(step + 1)
        ax1.plot(
            steps_range,
            agent_curve[: step + 1],
            label="SEVA (Agent)",
            color="#10B981",
            linewidth=3.2,
            marker="o",
            markersize=7,
        )

        if optuna_curve:
            ax1.plot(
                steps_range,
                optuna_curve[: step + 1],
                label="Optuna (TPE)",
                color="#3B82F6",
                linewidth=2.0,
                linestyle="--",
                marker="s",
                markersize=5,
            )
        if random_curve:
            ax1.plot(
                steps_range,
                random_curve[: step + 1],
                label="Random Search",
                color="#F59E0B",
                linewidth=1.8,
                linestyle=":",
                marker="^",
                markersize=5,
            )

        ax1.set_xlim(-0.3, max(1.0, n_trials - 0.7))
        ax1.set_ylim(0.87, 0.94)
        ax1.set_title(
            f"SEVA Closed-Loop Scientific Execution — Trial {step + 1}/{n_trials} (Dataset: credit-g)",
            fontsize=12.5,
            fontweight="bold",
            color="#F8FAFC",
            pad=10,
        )
        ax1.set_xlabel("Trial Budget Step", fontsize=10, color="#CBD5E1")
        ax1.set_ylabel("Best Val Macro-F1", fontsize=10, color="#CBD5E1")
        ax1.tick_params(colors="#94A3B8")
        for spine in ax1.spines.values():
            spine.set_color("#334155")
        ax1.grid(True, linestyle="--", alpha=0.25, color="#64748B")
        leg = ax1.legend(loc="lower right", facecolor="#0F172A", edgecolor="#334155", fontsize=9)
        for text in leg.get_texts():
            text.set_color("#F8FAFC")

        # Subplot 2: Scientific Lab Notebook Card (Bottom)
        ax2 = fig.add_axes([0.08, 0.05, 0.88, 0.32])
        ax2.set_facecolor("#1E293B")
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_color("#10B981" if current_t["was_new_best"] else "#334155")
            spine.set_linewidth(2.0 if current_t["was_new_best"] else 1.0)

        badge = "[NEW BEST RECORD]" if current_t["was_new_best"] else "[Exploration]"
        badge_color = "#34D399" if current_t["was_new_best"] else "#94A3B8"

        ax2.text(
            0.02,
            0.84,
            f"TRIAL #{step}  [{model_name}] — Val Macro-F1: {diag['val_macro_f1']:.4f}  |  Gap: {diag['overfit_gap']:+.4f}  |  {badge}",
            fontsize=10.0,
            fontweight="bold",
            color=badge_color,
            transform=ax2.transAxes,
        )

        clean_diag = (diag_text[:115] + "...") if len(diag_text) > 115 else diag_text
        clean_hyp = (hyp_text[:120] + "...") if len(hyp_text) > 120 else hyp_text
        clean_obs = (obs_text[:115] + "...") if len(obs_text) > 115 else obs_text

        ax2.text(
            0.02,
            0.58,
            f"[DIAGNOSIS]   {clean_diag}",
            fontsize=8.5,
            color="#E2E8F0",
            transform=ax2.transAxes,
        )
        ax2.text(
            0.02,
            0.32,
            f"[HYPOTHESIS]  {clean_hyp}",
            fontsize=8.5,
            color="#67E8F9",
            transform=ax2.transAxes,
        )
        ax2.text(
            0.02,
            0.08,
            f"[OBSERVED]    {clean_obs}",
            fontsize=8.5,
            color="#A7F3D0" if current_t.get("hypothesis_confirmed") else "#FCA5A5",
            transform=ax2.transAxes,
        )

        frame_path = tmp_frame_dir / f"frame_{step:02d}.png"
        fig.savefig(frame_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        img = Image.open(frame_path)
        frames.append(img)
        if current_t["was_new_best"]:
            frames.append(img)

    for _ in range(4):
        frames.append(frames[-1])

    gif_path = PLOTS_DIR / "autoresearcher_demo.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1300,
        loop=0,
    )

    for f in tmp_frame_dir.glob("*.png"):
        f.unlink()
    tmp_frame_dir.rmdir()

    print(f"Animated GIF successfully generated at: {gif_path}")
    return gif_path


if __name__ == "__main__":
    build_animated_gif()
