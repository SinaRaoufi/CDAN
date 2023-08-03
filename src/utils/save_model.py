import os
from datetime import datetime

import torch


def save_model(model, optimizer, loss, psnr, ssim, epoch, destination_dir, model_name='model.pt', device='cpu'):
    """
    Saves a PyTorch model to disk along with details of the training run.

    Args:
        model (nn.Module): A PyTorch model object to be saved.
        optimizer (optim.Optimizer): The optimizer used to train the model.
        loss (float): The validation loss value.
        psnr (float): The validation PSNR value.
        ssim (float): The validation SSIM value.
        epoch (int): The epoch where the model is saved.
        destination_dir (str): The directory where the saved model and details file will be stored.
        model_name (str, optional): The name of the saved model file. Defaults to 'model.pt'.
        device (str, optional): The device on which the model is trained. Defaults to 'cpu'.

    Returns:
        None
    """
    today_datetime = datetime.today().strftime('%Y-%m-%d')
    saved_model_dir = os.path.join(
        destination_dir, 'saved_models', today_datetime, device)

    os.makedirs(saved_model_dir, exist_ok=True)

    model_final_dir = os.path.join(saved_model_dir, model_name)
    details_final_dir = os.path.join(
        saved_model_dir, 'details_' + model_name + '.txt')

    # Save the entire model
    if device == 'mps':
        torch.save(model, model_final_dir)
    else:
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_final_dir)

    # Save details
    with open(details_final_dir, 'w') as f:
        f.write(f'Epoch: {epoch + 1}\n')
        f.write(f'PSNR: {psnr}\n')
        f.write(f'SSIM: {ssim}\n')
        f.write(f'Loss: {loss}\n')
        f.write(f'Optimizer: {optimizer}\n')
