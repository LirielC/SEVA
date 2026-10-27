"""Deploy the SEVA Static Showcase Space to Hugging Face."""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "Archiel89/SEVA"

HF_STATIC_README = """---
title: SEVA - Secure Experimentation and Validation Agent
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: mit
---

# SEVA: Secure Experimentation and Validation Agent

Interactive showcase of SEVA: Autonomous Diagnostic-Driven Machine Learning Pipeline Optimization for Mission-Critical Tabular Intelligence.
Visit the full codebase at https://github.com/LirielC/SEVA
"""


def deploy_static_space() -> None:
    api = HfApi(token=HF_TOKEN)
    root = Path(__file__).resolve().parents[1]
    hf_dir = root / "hf_space"
    hf_dir.mkdir(parents=True, exist_ok=True)

    # Copy plots into hf_space/artifacts/plots
    target_plots = hf_dir / "artifacts" / "plots"
    target_plots.mkdir(parents=True, exist_ok=True)

    src_plots = root / "artifacts" / "plots"
    for img_name in ["autoresearcher_demo.gif", "convergence_curves.png", "holdout_and_efficiency.png"]:
        src_file = src_plots / img_name
        if src_file.exists():
            shutil.copy2(src_file, target_plots / img_name)

    (hf_dir / "README.md").write_text(HF_STATIC_README, encoding="utf-8")

    print(f"Uploading files to Hugging Face Space: https://huggingface.co/spaces/{REPO_ID} ...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(hf_dir),
        commit_message="Deploy SEVA Interactive Showcase Space",
    )
    print(f"\nSUCCESS! Your Hugging Face Space is live at: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    deploy_static_space()
