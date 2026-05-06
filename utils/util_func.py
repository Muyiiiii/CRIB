import torch
import numpy as np
import os
import random
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def criterion_mape(y_true, y_pred, eps=1e-8):
    """
    Args:
        y_true (torch.Tensor): Ground truth values, shape (batch_size, ...)
        y_pred (torch.Tensor): Predicted values, shape (batch_size, ...)
        eps (float): Small constant to avoid division by zero
    Returns:
        torch.Tensor: MAPE (scalar)
    """
    # Ensure inputs are torch tensors
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    
    # Calculate absolute percentage error
    ape = torch.abs((y_true*10000 - y_pred*10000) / (y_true*10000 + eps*10000))
    
    # Take mean
    mape = torch.mean(ape)
    
    return mape