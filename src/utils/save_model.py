import os
from datetime import datetime

import torch


def save_model(model, destination_dir, model_name='model.pt', device='cpu'):
    today_datetime = datetime.today().strftime('%Y-%m-%d')
    saved_model_dir = os.path.join(
        destination_dir, 'saved_models', today_datetime, device)
    
    os.makedirs(saved_model_dir, exist_ok=True)

    final_dir = os.path.join(saved_model_dir, model_name)
    
    if device == 'mps':
        torch.save(model, final_dir)
    else:
        model_scripted = torch.jit.script(model)
        model_scripted.save(final_dir)
