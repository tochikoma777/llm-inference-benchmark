#!/usr/bin/env python3
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.scheduler_sim.request import generate_workload
from src.scheduler_sim.fcfs_scheduler import FCFSScheduler
from src.scheduler_sim.sjf_scheduler import SJFScheduler

def run_comparison():
    # 生成相同负载
    workload = generate_workload(num_requests=100, short_ratio=0.7, seed=42)
    
    # 运行 FCFS
    fcfs = FCFSScheduler()
    fcfs_results = fcfs.run(workload)
    
    # 运行 SJF（用相同负载，重新初始化）
    workload_sjf = generate_workload(num_requests=100, short_ratio=0.7, seed=42)
    sjf = SJFScheduler()
    sjf_results = sjf.run(workload_sjf)
    
    # 对比
    print("\n" + "="*60)
    print("Scheduler Comparison")
    print("="*60)
    print(f"{'Metric':<<25} {'FCFS':>15} {'SJF':>15}")
    print("-"*60)
    for key in ["avg_ttft_ms", "p99_ttft_ms", "avg_latency_ms", "throughput_tok_s"]:
        print(f"{key:<25} {fcfs_results[key]:>15.2f} {sjf_results[key]:>15.2f}")

    # ========== 新增：分组统计 ==========
    def print_group_stats(scheduler, label):
        short = [r for r in scheduler.completed if r.total_len < 500]
        long = [r for r in scheduler.completed if r.total_len >= 500]
        
        if short:
            short_ttfts = [s.ttft - s.arrival_time for s in short if s.ttft]
            print(f"\n{label} - Short Requests ({len(short)}): "
                  f"avg TTFT {sum(short_ttfts)/len(short_ttfts):.1f} ms, "
                  f"avg Latency {sum(s.finish_time - s.arrival_time for s in short)/len(short):.1f} ms")
        if long:
            long_ttfts = [l.ttft - l.arrival_time for l in long if l.ttft]
            print(f"{label} - Long  Requests ({len(long)}): "
                  f"avg TTFT {sum(long_ttfts)/len(long_ttfts):.1f} ms, "
                  f"avg Latency {sum(l.finish_time - l.arrival_time for l in long)/len(long):.1f} ms")

    print("\n--- Per-Group Analysis ---")
    print_group_stats(fcfs, "FCFS")
    print_group_stats(sjf, "SJF")
    # ==================================
    
    # 保存结果
    os.makedirs("data/results/scheduler", exist_ok=True)
    with open("data/results/scheduler/fcfs.json", "w") as f:
        json.dump(fcfs_results, f, indent=2)
    with open("data/results/scheduler/sjf.json", "w") as f:
        json.dump(sjf_results, f, indent=2)
    
    # 绘制对比图
    plot_comparison(fcfs_results, sjf_results)
    
    return fcfs_results, sjf_results

def plot_comparison(fcfs, sjf):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["avg_ttft_ms", "avg_latency_ms", "throughput_tok_s"]
    titles = ["Avg TTFT (ms)", "Avg Latency (ms)", "Throughput (tok/s)"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [fcfs[metric], sjf[metric]]
        colors = ["#1f77b4", "#ff7f0e"]
        ax.bar(["FCFS", "SJF"], values, color=colors, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Value", fontsize=10)
        
        # 标注数值
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.01, f"{v:.1f}", ha='center', fontsize=10)
    
    plt.suptitle("FCFS vs SJF Scheduling Policy Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/07_scheduler_comparison.png", dpi=150)
    plt.close()
    print("\n[Report] Saved to reports/figures/07_scheduler_comparison.png")

if __name__ == "__main__":
    run_comparison()