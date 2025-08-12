from typing import Optional

import torch

try:
    import torch_musa
except ModuleNotFoundError:
    torch_musa = None


def _is_musa():
    if torch_musa is None:
        return False
    else:
        return True


def get_device(local_rank: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        return (
            torch.cuda.current_device()
            if local_rank is None
            else torch.device("cuda", local_rank)
        )
    elif _is_musa():
        return (
            torch.musa.current_device()
            if local_rank is None
            else torch.device("musa", local_rank)
        )
    else:
        return torch.device("cpu")


def get_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif _is_musa():
        return "musa"
    else:
        return "cpu"


def get_torch_distributed_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    elif _is_musa():
        return "mccl"
    else:
        raise NotImplementedError(
            "No Accelerators(NV/MTT GPU) available"
        )
