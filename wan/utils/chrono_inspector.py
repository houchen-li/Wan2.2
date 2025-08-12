from time import perf_counter
from logging import info

from torch.cuda import synchronize

try:
    import torch_musa
    from torch_musa.core.device import synchronize
except ModuleNotFoundError:
    torch_musa = None


class ChronoInspector(object):
    def __init__(self, name:str="Block"):
        self.name = name

    def __enter__(self):
        synchronize()
        self.start_time:float = perf_counter()
        return self  # 可选：返回 self 以获取更多信息

    def __exit__(self, exc_type, exc_val, exc_tb):
        synchronize()
        end_time:float = perf_counter()
        info(f"[{self.name}] Elapsed time: {end_time - self.start_time:.2f} seconds")


__all__ = ["ChronoInspector"]
