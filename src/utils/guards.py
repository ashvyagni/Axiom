import numpy as np
import torch


def nan_guard(tensor, name="tensor"):
    """Check for NaN/Inf in tensor and replace if found."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    return tensor, {'has_nan': has_nan, 'has_inf': has_inf}


def clip_output(arr, min_val=0.0, max_val=1.0):
    """Clip numpy array to valid range and handle NaN/Inf."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=min_val)
    arr = np.clip(arr, min_val, max_val)
    return arr


def validate_output_shape(arr, expected_h=None, expected_w=None, name="output"):
    """Validate output array shape and dtype."""
    issues = []
    if arr.ndim not in (2, 3):
        issues.append(f"{name} has {arr.ndim} dims, expected 2 or 3")
    if arr.dtype != np.float32:
        issues.append(f"{name} dtype is {arr.dtype}, expected float32")
    if expected_h is not None and arr.shape[0] != expected_h:
        issues.append(f"{name} height {arr.shape[0]} != expected {expected_h}")
    if expected_w is not None and arr.shape[1] != expected_w:
        issues.append(f"{name} width {arr.shape[1]} != expected {expected_w}")
    return issues
