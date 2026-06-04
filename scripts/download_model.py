#!/usr/bin/env python3
from modelscope import snapshot_download
import os

CACHE_DIR = "/root/autodl-tmp/modelscope"
MODEL_ID = "qwen/Qwen2.5-7B-Instruct"

print(f"[*] Downloading {MODEL_ID} ...")
print(f"[*] Cache dir: {CACHE_DIR}")

model_dir = snapshot_download(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    local_files_only=False
)

print(f"\n[✓] Model downloaded to: {model_dir}")
print(f"[✓] Use this path in benchmarks: {model_dir}")
