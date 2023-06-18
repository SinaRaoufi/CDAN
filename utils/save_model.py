import os
from datetime import datetime

import torch


SAVE_DIR_ROOT = "/home/novin/Desktop/Final_Essay/Diabetic-Retinopathy-Classification"


def save_model(model, model_name='model.pt'):
    today_datetime = datetime.today().strftime('%Y-%m-%d')
    saved_model_dir = os.path.join(
        SAVE_DIR_ROOT, 'saved_models', today_datetime)
    os.makedirs(saved_model_dir, exist_ok=True)

    torch.save(model, os.path.join(saved_model_dir, model_name))