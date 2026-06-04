import random
from dataclasses import dataclass
from typing import List

@dataclass
class Request:
    id: int
    arrival_time: float          # 到达时间（ms）
    prompt_len: int              # 输入 token 数
    output_len: int              # 期望输出 token 数
    actual_output_len: int = 0   # 实际生成 token 数（运行后填充）
    start_time: float = None     # 开始执行时间
    finish_time: float = None    # 完成时间
    ttft: float = None           # 首 token 时间（prefill 完成）
    
    @property
    def total_len(self):
        return self.prompt_len + self.output_len
    
    @property
    def is_short(self):
        return self.total_len < 500  # 用于 SJF 分类

def generate_workload(
    num_requests: int = 100,
    short_ratio: float = 0.7,    # 70% 短请求
    seed: int = 42
) -> List[Request]:
    """生成混合负载：短请求（prompt<<256, output<<128）+ 长请求（prompt>1024, output>512）"""
    random.seed(seed)
    requests = []
    
    for i in range(num_requests):
        arrival = i * random.uniform(50, 150)  # 随机到达间隔 50~150ms
        
        if random.random() < short_ratio:
            # 短请求：模拟 FAQ/闲聊
            prompt = random.randint(50, 256)
            output = random.randint(20, 128)
        else:
            # 长请求：模拟文档生成/代码补全
            prompt = random.randint(512, 2048)
            output = random.randint(256, 1024)
        
        requests.append(Request(
            id=i,
            arrival_time=arrival,
            prompt_len=prompt,
            output_len=output
        ))
    
    return requests