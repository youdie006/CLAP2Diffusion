"""
Safe dataset wrapper with simple caching
Preserves original dataset functionality while adding optimization
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any
import copy


class DatasetWrapper(Dataset):
    """Wrapper to add caching to existing dataset without modification"""
    
    def __init__(self, original_dataset, cache_enabled=True, cache_size=1000):
        self.dataset = original_dataset
        self.cache_enabled = cache_enabled
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if not self.cache_enabled:
            return self.dataset[idx]
        
        # Check cache
        if idx in self.cache:
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # Load from original dataset
        data = self.dataset[idx]
        
        # Deep copy to prevent cache pollution from augmentation
        data_copy = copy.deepcopy(data)
        
        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = data_copy
            self.access_count[idx] = 1
        else:
            # LRU eviction
            if self.access_count:
                min_idx = min(self.access_count, key=self.access_count.get)
                del self.cache[min_idx]
                del self.access_count[min_idx]
                self.cache[idx] = data_copy
                self.access_count[idx] = 1
        
        return data
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_count.clear()