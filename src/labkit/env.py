"""The machine a notebook runs on: the Hub login, the seeds, the GPU."""

__all__ = ['login_hf', 'empty_cache', 'gpu_report', 'describe_env', 'set_determinism', 'installed_commit']

import gc
import os
import subprocess


def login_hf(token=None, required=True):
    """Authenticate against the Hugging Face Hub.

    Tries, in order: an explicit `token`, the `HF_TOKEN` environment variable,
    and the Colab secret named `HF_TOKEN`. Writing needs a token with write
    scope; reading the public repos needs none, so a failure here is only fatal
    if you intend to push — pass `required=False` to read anonymously without one.
    """
    from huggingface_hub import login

    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:                                  # noqa: BLE001 — not on Colab
            pass
    if token is None:
        if not required:
            print("No Hugging Face token: reading the public repos anonymously.")
            return False
        raise RuntimeError(
            "No token found. Pass login_hf(token=...), set HF_TOKEN, or add an "
            "HF_TOKEN secret in Colab (key icon in the left sidebar)."
        )
    login(token=token)
    print("Logged in to the Hugging Face Hub.")
    return True


def empty_cache(report=True):
    """Drop unreferenced tensors and return the freed GPU memory to the driver."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    if report:
        print(gpu_report())


def gpu_report():
    """One line of `nvidia-smi` memory use, or a note if there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or "nvidia-smi returned nothing"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "no NVIDIA GPU visible"


def describe_env():
    """Print the versions that decide whether a model cell will run."""
    import platform
    print(f"python       {platform.python_version()}")
    for mod in ("torch", "transformers", "datasets", "numpy", "nemo"):
        try:
            m = __import__(mod)
            print(f"{mod:<12} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{mod:<12} —")
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
        print(f"device       {dev}")
    except ImportError:
        pass
    print(f"gpu memory   {gpu_report()}")


def set_determinism(seed=0, strict=False):
    """Seed Python, NumPy and torch. `strict` also asks CUDA for deterministic kernels.

    Deterministic attention kernels are slower and a few are unavailable in
    fp16, so `strict=False` is the default: the seeds make a run reproducible
    to the noise floor of the GPU's own reduction order, which for the numbers
    reported here is below the third decimal of WER.
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if strict:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass
    return seed


def installed_commit(package="hitasr"):
    """The git commit the distribution providing `package` was installed from (`pip install git+...`), shortened;
    `"local"` for an editable or plain install, `"unknown"` when it is not installed."""
    import importlib.metadata as md
    import json
    for dist in md.packages_distributions().get(package, []):
        try:
            info = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
        except (md.PackageNotFoundError, ValueError):
            continue
        commit = info.get("vcs_info", {}).get("commit_id")
        return commit[:8] if commit else "local"
    return "unknown"
