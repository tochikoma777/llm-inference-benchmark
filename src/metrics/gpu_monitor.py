import threading
import time
import subprocess

class GPUMonitor:
    """后台线程监控GPU显存和利用率"""
    def __init__(self, interval=0.1, device_id=0):
        self.interval = interval
        self.device_id = device_id
        self.data = []
        self._stop_event = threading.Event()
        self._thread = None
    
    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "-i", str(self.device_id)
                    ],
                    capture_output=True, text=True, check=True
                )
                line = result.stdout.strip()
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    self.data.append({
                        "timestamp": time.perf_counter(),
                        "memory_used_mb": int(parts[0]),
                        "memory_total_mb": int(parts[1]),
                        "gpu_util": int(parts[2]),
                    })
            except Exception as e:
                print(f"[GPUMonitor] Error: {e}")
            time.sleep(self.interval)
    
    def start(self):
        self.data = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def get_peak_memory_mb(self):
        if not self.data:
            return 0
        return max(d["memory_used_mb"] for d in self.data)
    
    def get_avg_gpu_util(self):
        if not self.data:
            return 0
        return sum(d["gpu_util"] for d in self.data) / len(self.data)
    
    def summary(self):
        return {
            "peak_memory_mb": self.get_peak_memory_mb(),
            "avg_gpu_util": self.get_avg_gpu_util(),
            "samples": len(self.data)
        }
