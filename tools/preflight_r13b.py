#!/usr/bin/env python3
"""Check that the r13b (Mistral-7B) sweep will be able to start.

r13b is the only sweep with requirements beyond torch + transformers, and it
runs last in check_all.sh --experiments -- many hours in. This script answers
"will it start?" in a few seconds so a missing dependency is found now rather
than after the rest of the queue has finished.

Safe to run while a sweep is in progress: it never imports CUDA, never
allocates on the GPU, and downloads nothing: it only checks that modules
import, that the model repo is reachable, and how much of it is already in the
local Hugging Face cache.

    python3 tools/preflight_r13b.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys

MODEL = "mistralai/Mistral-7B-v0.1"   # as set in run_mistral_task1.py
APPROX_DOWNLOAD_GB = 15.0             # fp16 safetensors shards
OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
problems = 0
warnings = 0


def line(tag: str, what: str, detail: str = "") -> None:
    print(f"[{tag}] {what}" + (f"   {detail}" if detail else ""))


def need(mod: str, why: str) -> bool:
    global problems
    try:
        m = importlib.import_module(mod)
    except Exception as e:                                   # noqa: BLE001
        problems += 1
        line(BAD, f"{mod:<16} MISSING", f"({why}) -- {type(e).__name__}")
        return False
    line(OK, f"{mod:<16} {getattr(m, '__version__', '?')}", why)
    return True


print(f"\npreflight for r13b -- {MODEL}\n" + "=" * 62)

# ---------------------------------------------------------------- imports --
need("torch", "everything")
need("transformers", "model + tokenizer")
have_bnb = need("bitsandbytes", "4-bit quantisation: load_in_4bit=True")
have_acc = need("accelerate", 'required by device_map="auto"')

if not (have_bnb and have_acc):
    miss = " ".join(m for m, h in (("bitsandbytes", have_bnb),
                                   ("accelerate", have_acc)) if not h)
    print(f"\n  install with:  conda install -c conda-forge {miss}")
    print(f"  or:            pip install {miss}")

# ------------------------------------------------------------------ cache --
try:
    from huggingface_hub import constants, snapshot_download  # noqa: F401
    cache = constants.HF_HUB_CACHE
except Exception:                                            # noqa: BLE001
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

repo_dir = os.path.join(cache, "models--" + MODEL.replace("/", "--"))
cached_gb = 0.0
if os.path.isdir(repo_dir):
    for root, _, files in os.walk(repo_dir):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                try:
                    cached_gb += os.path.getsize(p)
                except OSError:
                    pass
    cached_gb /= 1024 ** 3

if cached_gb >= APPROX_DOWNLOAD_GB * 0.9:
    line(OK, f"{'model cache':<16} {cached_gb:.1f} GB present", "no download needed")
    still_to_fetch = 0.0
elif cached_gb > 0:
    warnings += 1
    line(WARN, f"{'model cache':<16} {cached_gb:.1f} GB partial",
         f"~{APPROX_DOWNLOAD_GB - cached_gb:.0f} GB still to download")
    still_to_fetch = APPROX_DOWNLOAD_GB - cached_gb
else:
    warnings += 1
    line(WARN, f"{'model cache':<16} empty",
         f"~{APPROX_DOWNLOAD_GB:.0f} GB will be downloaded on first use")
    still_to_fetch = APPROX_DOWNLOAD_GB

# ------------------------------------------------------------------- disk --
free_gb = shutil.disk_usage(os.path.dirname(repo_dir) if os.path.isdir(
    os.path.dirname(repo_dir)) else os.path.expanduser("~")).free / 1024 ** 3
if free_gb < still_to_fetch + 2:
    problems += 1
    line(BAD, f"{'free disk':<16} {free_gb:.1f} GB",
         f"need ~{still_to_fetch + 2:.0f} GB on the cache volume")
else:
    line(OK, f"{'free disk':<16} {free_gb:.1f} GB", f"cache: {cache}")

# ------------------------------------------------------- repo reachability --
try:
    from huggingface_hub import HfApi
    info = HfApi().model_info(MODEL)
    gated = getattr(info, "gated", False)
    if gated:
        problems += 1
        line(BAD, f"{'repo access':<16} GATED",
             "accept the terms on the model page, then `huggingface-cli login`")
    else:
        line(OK, f"{'repo access':<16} public", "no token needed")
except ImportError:
    warnings += 1
    line(WARN, f"{'repo access':<16} unchecked", "huggingface_hub not importable")
except Exception as e:                                       # noqa: BLE001
    warnings += 1
    line(WARN, f"{'repo access':<16} unchecked",
         f"{type(e).__name__} -- offline? this is only fatal if the cache is empty")

# ------------------------------------------------------------------ verdict --
print("=" * 62)
if problems:
    print(f"{problems} blocking problem(s): r13b would FAIL when it is reached.")
    print("Everything else in the queue is unaffected -- a failing sweep does not")
    print("stop the others. Fix these, then re-run just that one:")
    print("    ./check_all.sh --experiments-only --only r13b")
    sys.exit(1)
print("r13b can start." + (f"  ({warnings} note(s) above.)" if warnings else ""))
sys.exit(0)
