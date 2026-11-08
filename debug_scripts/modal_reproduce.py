from pathlib import Path
import subprocess

import modal

APP = modal.App("specsafe-reproduce-s7")

if modal.is_local():
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    SPECSAFE_DIR = PROJECT_DIR / "specsafe"
    VLLM_DIR = PROJECT_DIR / "vllm"
else:
    SPECSAFE_DIR = Path("/root/specsafe")
    VLLM_DIR = Path("/root/vllm")


MODEL_VOLUME = modal.Volume.from_name(
    "qwen3-8-27b-fp8", create_if_missing=True
)
RESULTS_VOLUME = modal.Volume.from_name(
    "specsafe-results", create_if_missing=True
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.32.0",
        "huggingface_hub",
        extra_options="--torch-backend=cu130",
    )
    .add_local_dir(SPECSAFE_DIR, remote_path="/root/specsafe", copy=True)
    .add_local_dir(
        VLLM_DIR,
        remote_path="/root/vllm",
        copy=True,
        ignore=[".git", ".venv", "build", "__pycache__"],
    )
    # Do not add /root/vllm here: it would override the clean vLLM 0.32.0 wheel
    # with your checkout's source code.
    .env({"PYTHONPATH": "/root/specsafe"})
)


MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"
MODEL_DIR = "/models/qwen3.8-27b-fp8"
MODEL_READY_FILE = f"{MODEL_DIR}/.download-complete"


@APP.function(
    image=image,
    timeout=2 * 60 * 60,
    volumes={"/models": MODEL_VOLUME},
)
def populate_model() -> None:
    from pathlib import Path
    from huggingface_hub import snapshot_download

    if Path(MODEL_READY_FILE).exists():
        return

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
    )
    Path(MODEL_READY_FILE).touch()
    MODEL_VOLUME.commit()


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=4 * 60 * 60,
    volumes={"/models": MODEL_VOLUME, "/results": RESULTS_VOLUME},
)
def reproduce(
    case: str = "S7",
    k: int = 5,
    n: int = 8,
    length: int = 192,
    selector: str = "first8",
) -> None:
    subprocess.run(
        [
            "python",
            "/root/vllm/debug_scripts/reproduce.py",
            "--case", case,
            "--k", str(k),
            "--n", str(n),
            "--len", str(length),
            "--selector", selector,
            "--model", MODEL_DIR,
            "--out", "/results/drift_min",
        ],
        check=True,
    )
    RESULTS_VOLUME.commit()


@APP.local_entrypoint()
def main() -> None:
    populate_model.remote()
    reproduce.remote()
