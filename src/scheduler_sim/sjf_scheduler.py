from typing import List
from .base_scheduler import BaseScheduler
from .request import Request

class SJFScheduler(BaseScheduler):
    """短作业优先：降低平均等待时间，但可能饿死长请求"""
    
    def select_requests(self, waiting: List[Request]) -> List[Request]:
        if not waiting:
            return []
        
        # SJF：按总长度排序，短的优先
        batch_size = min(4, len(waiting))
        return sorted(waiting, key=lambda r: r.total_len)[:batch_size]