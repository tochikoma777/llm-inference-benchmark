from typing import List
from .base_scheduler import BaseScheduler
from .request import Request

class FCFSScheduler(BaseScheduler):
    """先来先服务：公平，但短请求可能被长请求阻塞"""
    
    def select_requests(self, waiting: List[Request]) -> List[Request]:
        if not waiting:
            return []
        
        # FCFS：只取最早到达的一个请求（简化版，实际 vLLM 会 batch）
        # 为了对比效果，这里允许 batch 最多 4 个请求
        batch_size = min(4, len(waiting))
        return sorted(waiting, key=lambda r: r.arrival_time)[:batch_size]