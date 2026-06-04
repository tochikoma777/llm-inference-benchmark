from typing import Dict, Set

class MemoryPool:
    """简化版 PagedAttention 内存池：固定大小 blocks，按需分配"""
    
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = total_blocks
        self.allocated: Dict[int, int] = {}  # request_id -> num_blocks
    
    def allocate(self, request_id: int, num_tokens: int) -> bool:
        """为请求分配 KV Cache blocks，成功返回 True"""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if blocks_needed > self.free_blocks:
            return False  # OOM，需要 swap 或等待
        
        self.allocated[request_id] = blocks_needed
        self.free_blocks -= blocks_needed
        return True
    
    def release(self, request_id: int):
        """释放请求的 KV Cache"""
        if request_id in self.allocated:
            self.free_blocks += self.allocated[request_id]
            del self.allocated[request_id]
    
    def can_allocate(self, num_tokens: int) -> bool:
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return blocks_needed <= self.free_blocks
    
    @property
    def utilization(self) -> float:
        return (self.total_blocks - self.free_blocks) / self.total_blocks