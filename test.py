import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from PIL import Image
from tqdm import tqdm
import numpy as np
from decouple import config

from dataset import CDANDataset


def evaluate(model, optimizer, criterion, data_loader: dict, save_dir: str, device='cpu'):
    model.to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    with torch.no_grad():
        test_loss = 0.0
        test_psnr = 0.0
        test_ssim = 0.0
        test_lpips = 0.0
        model.eval()
        for inputs, targets in tqdm(data_loader, desc='Testing...'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_psnr += psnr(outputs, targets)
            test_ssim += ssim(outputs, targets)
            test_lpips += lpips(outputs, targets)

        test_loss = test_loss / len(data_loader)
        test_psnr = test_psnr / len(data_loader)
        test_ssim = test_ssim / len(data_loader)
        test_lpips = test_lpips / len(data_loader)

        print(
            f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}, Test LIPIS: {test_lpips:.4f}')

        for i, output_image in enumerate(outputs):
            output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)

            output_path = os.path.join(save_dir, f'output_{i + 1}.png')

            output_image.save(output_path)


if __name__ == '__main__':
    INPUT_SIZE = 200
    DATASET_DIR_ROOT = config('DATASET_DIR_ROOT')
    MODEL_PATH = 'best.pt'
    SAVE_DIR_ROOT = config('SAVE_DIR_ROOT')
    BATCH_SIZE = 16

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'

    test_transforms = transforms.Compose([
        transforms.Resize((400, 600)),
        transforms.ToTensor(),
    ])

    test_dataset = CDANDataset(
        os.path.join(DATASET_DIR_ROOT, 'test', 'low'),
        os.path.join(DATASET_DIR_ROOT, 'test', 'high'),
        test_transforms,
        test_transforms
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    
    model = torch.load(MODEL_PATH)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    evaluate(model, optimizer, criterion, test_dataloader, SAVE_DIR_ROOT, device)
