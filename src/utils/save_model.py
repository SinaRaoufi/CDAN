import os
from datetime import datetime

import torch


def save_model(model, optimizer, loss, psnr, ssim, epoch, destination_dir, model_name='model', device='cpu'):
    """
    Saves model parameters to disk along with details of the training run.

    Args:
        model (nn.Module): A PyTorch model object to be saved.
        optimizer (optim.Optimizer): The optimizer used to train the model.
        loss (float): The validation loss value.
        psnr (float): The validation PSNR value.
        ssim (float): The validation SSIM value.
        epoch (int): The epoch where the model is saved.
        destination_dir (str): The directory where the saved model and details file will be stored.
        model_name (str, optional): The name of the saved model file. Defaults to 'model'.
        device (str, optional): The device on which the model is trained. Defaults to 'cpu'.

    Returns:
        None
    """
    today_datetime = datetime.today().strftime('%Y-%m-%d')
    saved_model_dir = os.path.join(
        destination_dir, 'saved_models', today_datetime, device, model_name)
    os.makedirs(saved_model_dir, exist_ok=True)
    model_final_dir = os.path.join(saved_model_dir, model_name + '.pt')
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'PSNR': psnr,
        'SSIM': ssim
        }, model_final_dir)
