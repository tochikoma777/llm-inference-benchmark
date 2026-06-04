from abc import ABC, abstractmethod
from typing import List, Dict
from .request import Request
from .gpu import GPU, GPUConfig
from .memory_pool import MemoryPool

class BaseScheduler(ABC):
    """调度器基类：定义 vLLM-like 调度循环"""
    
    def __init__(self, gpu_config: GPUConfig = None):
        self.gpu = GPU(gpu_config)
        # 基于 GPU 显存计算 block 数量（24GB 中约 20GB 给 KV Cache）
        kv_cache_mb = 20000
        self.memory = MemoryPool(
            total_blocks=kv_cache_mb // 16,  # 16 tokens per block
            block_size=16
        )
        self.current_time = 0.0
        self.completed: List[Request] = []
        self.metrics: Dict = {}
    
    @abstractmethod
    def select_requests(self, waiting: List[Request]) -> List[Request]:
        """从等待队列中选择进入 running 的请求"""
        pass
    
    def run(self, requests: List[Request]) -> Dict:
        """主调度循环：模拟 vLLM 的调度行为"""
        waiting = []      # 等待队列
        running = []      # 正在执行（prefill 或 decode）
        finished = []     # 已完成
        
        # 按到达时间排序
        requests = sorted(requests, key=lambda r: r.arrival_time)
        next_arrival_idx = 0
        
        while len(finished) < len(requests):
            # 1. 处理新到达请求
            while (next_arrival_idx < len(requests) and 
                   requests[next_arrival_idx].arrival_time <= self.current_time):
                waiting.append(requests[next_arrival_idx])
                next_arrival_idx += 1
            
            # 2. 调度策略：选择请求进入 running
            selected = self.select_requests(waiting)
            for req in selected:
                waiting.remove(req)
                # 尝试分配 KV Cache
                if self.memory.allocate(req.id, req.prompt_len + req.output_len):
                    req.start_time = self.current_time
                    running.append(req)
                else:
                    # 内存不足，放回 waiting（vLLM 会 swap，这里简化）
                    waiting.insert(0, req)
            
            # 3. 执行一步：所有 running 请求推进一个时间片
            if running:
                time_slice = 10.0  # 10ms 时间片（模拟 vLLM 的调度粒度）
                self._execute_step(running, finished, time_slice)
                self.current_time += time_slice
            else:
                # 无请求运行，跳转到下一个到达时间
                if next_arrival_idx < len(requests):
                    self.current_time = requests[next_arrival_idx].arrival_time
                else:
                    break
        
        self.completed = finished
        return self._compute_metrics()
    
    # def _execute_step(self, running: List[Request], finished: List[Request], dt: float):
    #     """执行一个时间片：推进 running 请求的 prefill/decode"""
    #     for req in running[:]:  # 复制列表，因为可能删除
    #         if req.ttft is None:
    #             # 仍在 prefill 阶段
    #             prefill_done = self.gpu.prefill_time(req.prompt_len)
    #             if self.current_time + dt >= req.start_time + prefill_done:
    #                 req.ttft = req.start_time + prefill_done
    #         else:
    #             # 在 decode 阶段
    #             decode_done = self.gpu.decode_time(req.output_len)
    #             elapsed = self.current_time - req.ttft
    #             if elapsed + dt >= decode_done:
    #                 # 请求完成
    #                 req.finish_time = self.current_time + dt
    #                 req.actual_output_len = req.output_len
    #                 finished.append(req)
    #                 running.remove(req)
    #                 self.memory.release(req.id)
    def _execute_step(self, running: List[Request], finished: List[Request], dt: float):
        """执行一个时间片：推进 running 请求的 prefill/decode，
        prefill 完成后立即用剩余时间推进 decode（更接近真实 vLLM 行为）"""
        for req in running[:]:
            remaining_time = dt

            # 1. 如果还在 prefill 阶段
            if req.ttft is None:
                prefill_needed = self.gpu.prefill_time(req.prompt_len)
                elapsed_prefill = self.current_time - req.start_time
                prefill_left = prefill_needed - elapsed_prefill

                if prefill_left > remaining_time:
                    # prefill 未完成，整个时间片都用于 prefill
                    continue
                else:
                    # prefill 在本时间片内完成
                    req.ttft = self.current_time + prefill_left
                    remaining_time -= prefill_left
                    # 不 continue，继续进入 decode 逻辑

            # 2. 已经进入 decode 阶段（或刚完成 prefill）
            if req.ttft is not None:
                decode_needed = self.gpu.decode_time(req.output_len)
                # 从 TTFT 开始到本时间片结束，已经 decode 的时间
                decode_elapsed = (self.current_time + dt) - req.ttft

                if decode_elapsed >= decode_needed:
                    # 请求完成
                    req.finish_time = self.current_time + dt
                    req.actual_output_len = req.output_len
                    finished.append(req)
                    running.remove(req)
                    self.memory.release(req.id)
                # 否则 decode 未完成，等待下一轮    
    
    def _compute_metrics(self) -> Dict:
        """计算调度性能指标"""
        if not self.completed:
            return {}
        
        ttfts = [r.ttft - r.arrival_time for r in self.completed if r.ttft]
        latencies = [r.finish_time - r.arrival_time for r in self.completed]
        
        total_time = max(r.finish_time for r in self.completed)
        total_tokens = sum(r.actual_output_len for r in self.completed)
        
        return {
            "scheduler": self.__class__.__name__,
            "num_requests": len(self.completed),
            "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
            "p99_ttft_ms": sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts) > 1 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "throughput_tok_s": total_tokens / (total_time / 1000),
            "gpu_util_avg": sum(r.actual_output_len for r in self.completed) / 
                           (total_time / 1000 * self.gpu.config.decode_tok_per_ms * 1000),
            "memory_util_avg": self.memory.utilization,
        }