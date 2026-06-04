#!/usr/bin/env python3
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_results():
    records = []
    
    # Stage 2: concurrency data (throughput scan)
    for f in glob.glob("data/results/concurrency/*.json"):
        with open(f) as fh:
            d = json.load(fh)
            records.append({
                "backend": d.get("backend"),
                "concurrency": d.get("concurrency"),
                "throughput_tok_s": d.get("throughput", 0),           # ← 必须映射
                "avg_gpu_util_pct": d.get("avg_gpu_util", 0),        # ← 必须映射
                "total_time_ms": d.get("total_time", 0) * 1000,
                "peak_memory_mb": d.get("peak_memory_mb", 0),
                "prompt_len": None,
                "output_len": None,
            })
    
    # Stage 3: latency breakdown (TTFT/TPOT)
    for f in glob.glob("data/results/latency/*.json"):
        with open(f) as fh:
            data = json.load(fh)
            items = data if isinstance(data, list) else [data]
            for item in items:
                records.append({
                    "backend": item.get("backend"),
                    "concurrency": None,
                    "throughput_tok_s": item.get("throughput_tok_s", item.get("throughput", 0)),
                    "avg_gpu_util_pct": item.get("avg_gpu_util_pct", item.get("avg_gpu_util", 0)),
                    "total_time_ms": item.get("total_time_ms", item.get("total_time", 0) * 1000),
                    "peak_memory_mb": item.get("peak_memory_mb", 0),
                    "prompt_len": item.get("prompt_len"),
                    "output_len": item.get("output_len"),
                })
    
    # Stage 3: memory scan
    for f in glob.glob("data/results/memory/*.json"):
        with open(f) as fh:
            data = json.load(fh)
            items = data if isinstance(data, list) else [data]
            for item in items:
                records.append({
                    "backend": item.get("backend"),
                    "concurrency": None,
                    "throughput_tok_s": item.get("throughput_tok_s", item.get("throughput", 0)),
                    "avg_gpu_util_pct": item.get("avg_gpu_util_pct", item.get("avg_gpu_util", 0)),
                    "total_time_ms": item.get("total_time_ms", item.get("total_time", 0) * 1000),
                    "peak_memory_mb": item.get("peak_memory_mb", 0),
                    "prompt_len": item.get("prompt_len"),
                    "output_len": item.get("output_len"),
                })
    
    df = pd.DataFrame(records)
    print(f"[Report] Loaded {len(df)} records, columns: {list(df.columns)}")
    return df

def plot_ttft_vs_prompt_length(df, outdir):
    sub = df[df['output_len'] == 1].sort_values('prompt_len')
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in sub['backend'].unique():
        bsub = sub[sub['backend'] == backend]
        ax.plot(bsub['prompt_len'], bsub['total_time_ms'], marker='o', 
                label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Prompt Length (tokens)', fontsize=12)
    ax.set_ylabel('TTFT (ms)', fontsize=12)
    ax.set_title('TTFT vs Prompt Length (output_len=1)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '01_ttft_vs_prompt_length.png'), dpi=150)
    plt.close()
    print("[Report] 01_ttft_vs_prompt_length.png")

def plot_throughput_vs_concurrency(df, outdir):
    sub = df[df['concurrency'].notna()].sort_values('concurrency')
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in sub['backend'].unique():
        bsub = sub[sub['backend'] == backend]
        ax.plot(bsub['concurrency'], bsub['throughput_tok_s'], marker='o',
                label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Throughput vs Concurrency', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '02_throughput_vs_concurrency.png'), dpi=150)
    plt.close()
    print("[Report] 02_throughput_vs_concurrency.png")

def plot_memory_vs_sequence_length(df, outdir):
    sub = df[df['prompt_len'].notna()].sort_values('prompt_len')
    if sub.empty or 'peak_memory_mb' not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in sub['backend'].unique():
        bsub = sub[sub['backend'] == backend]
        ax.plot(bsub['prompt_len'], bsub['peak_memory_mb'], marker='o',
                label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax.set_ylabel('Peak GPU Memory (MB)', fontsize=12)
    ax.set_title('GPU Memory vs Sequence Length', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '03_memory_vs_sequence_length.png'), dpi=150)
    plt.close()
    print("[Report] 03_memory_vs_sequence_length.png")

def plot_gpu_utilization(df, outdir):
    sub = df[df['concurrency'].notna()].sort_values('concurrency')
    if sub.empty or 'avg_gpu_util_pct' not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in sub['backend'].unique():
        bsub = sub[sub['backend'] == backend]
        ax.plot(bsub['concurrency'], bsub['avg_gpu_util_pct'], marker='o',
                label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Average GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization: vLLM vs Naive', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '04_gpu_utilization.png'), dpi=150)
    plt.close()
    print("[Report] 04_gpu_utilization.png")

def plot_prefix_cache(outdir):
    files = glob.glob("data/results/prefix_cache/*.json")
    if not files:
        return
    data = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
            d['file'] = os.path.basename(f)
            data.append(d)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = ['Round 1', 'Round 2', 'Round 3']
    x = np.arange(len(rounds))
    width = 0.35
    
    for i, d in enumerate(data):
        times = [d['round1_time'], d['round2_time'], d['round3_time']]
        label = 'Cache ON' if d['enable_prefix_caching'] else 'Cache OFF'
        color = '#2ca02c' if d['enable_prefix_caching'] else '#d62728'
        ax.bar(x + i*width, times, width, label=label, color=color, alpha=0.8)
    
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Prefix Caching Speedup (Multi-turn Dialogue)', fontsize=14)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(rounds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '05_prefix_cache.png'), dpi=150)
    plt.close()
    print("[Report] 05_prefix_cache.png")

def plot_batching_comparison(df, outdir):
    sub = df[df['concurrency'].notna()].sort_values('concurrency')
    if sub.empty or 'total_time_ms' not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in sub['backend'].unique():
        bsub = sub[sub['backend'] == backend]
        ax.plot(bsub['concurrency'], bsub['total_time_ms']/1000, marker='o',
                label=backend.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Total Completion Time (s)', fontsize=12)
    ax.set_title('Static vs Continuous Batching: Completion Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, '06_batching_comparison.png'), dpi=150)
    plt.close()
    print("[Report] 06_batching_comparison.png")

def main():
    outdir = "reports/figures"
    os.makedirs(outdir, exist_ok=True)
    
    df = load_all_results()
    if df.empty:
        print("[Error] No data found in data/results/")
        return
    
    print(f"[Report] Loaded {len(df)} records")
    
    plot_ttft_vs_prompt_length(df, outdir)
    plot_throughput_vs_concurrency(df, outdir)
    plot_memory_vs_sequence_length(df, outdir)
    plot_gpu_utilization(df, outdir)
    plot_prefix_cache(outdir)
    plot_batching_comparison(df, outdir)
    
    print(f"\n[Report] All figures saved to {outdir}/")

if __name__ == "__main__":
    main()