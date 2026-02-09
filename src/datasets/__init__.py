"""Dataset loaders for causal discovery experiments."""

from .credit_approval import load_credit_approval, get_dataset_info as get_credit_info
from .wine_quality import load_wine_quality, get_dataset_info as get_wine_info

__all__ = [
    'load_credit_approval',
    'load_wine_quality',
    'get_credit_info',
    'get_wine_info'
]
