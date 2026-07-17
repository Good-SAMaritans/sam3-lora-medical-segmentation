"""
Pick the best available torch device so the same code runs on 
Mac (MPS), the cluster (CUDA), and CI (CPU) unchanged
"""

import torch

def get_device():
    """
    return string not torch device object
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
