import os
from datetime import datetime

from PIL import Image
import numpy as np


def save_output_images(outputs, save_dir, epoch, model_name, device, is_best_result=False):
    """Save output images to the specified directory.

    Args:
        outputs (torch.Tensor): A Tensor of output images to be saved.
        save_dir (str): The base directory where the output images will be saved.
        epoch (int): The current epoch number.
        model_name (str): The name of the model.
        device (str): The device on which the model is trained (e.g., 'cpu', 'mps', 'cuda').
        is_best_result (bool, optional): Flag indicating whether the outputs represent the best results.
            Defaults to False.

    Returns:
        None
    """

    today_datetime = datetime.today().strftime('%Y-%m-%d')
    save_dir = os.path.join(
        save_dir, 'saved_models', today_datetime, device, model_name, 'output_images')
    if is_best_result:
        save_dir = os.path.join(save_dir, 'best_outputs')
    else:
        save_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(save_dir, exist_ok=True)

    for i, output_image in enumerate(outputs):
        output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)

        output_path = os.path.join(save_dir, f'output_{i + 1}.png')

        output_image.save(output_path)
