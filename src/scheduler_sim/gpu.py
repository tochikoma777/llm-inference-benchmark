from dataclasses import dataclass

@dataclass
class GPUConfig:
    """GPU 硬件参数（基于 RTX 3090 实测数据校准）"""
    prefill_tok_per_ms: float = 2.5    # prefill 吞吐：2500 tokens/s
    decode_tok_per_ms: float = 0.05    # decode 吞吐：50 tokens/s
    memory_mb: int = 24000             # 总显存 24GB
    kv_cache_per_token_mb: float = 0.5  # 每个 token 的 KV Cache 约 0.5MB（7B 模型 fp16）

class GPU:
    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.current_time = 0.0
    
    def prefill_time(self, prompt_len: int) -> float:
        """计算 prefill 阶段耗时（ms）"""
        return prompt_len / self.config.prefill_tok_per_ms
    
    def decode_time(self, output_len: int) -> float:
        """计算 decode 阶段耗时（ms）"""
        return output_len / self.config.decode_tok_per_ms
    
    def kv_cache_needed(self, seq_len: int) -> float:
        """计算某序列需要的 KV Cache 显存（MB）"""
        return seq_len * self.config.kv_cache_per_token_mb