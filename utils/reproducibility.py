import random

import torch
import numpy as np

def set_seed_and_cudnn(seed_value=42):
  """
  Sets the random seed for reproducibility and configures cuDNN for determinism.

  Args:
      seed_value (int, optional): The seed value to use for reproducibility. Defaults to 42.
  """
  
  # Set random seeds
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)

  # Configure cuDNN for determinism
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = True