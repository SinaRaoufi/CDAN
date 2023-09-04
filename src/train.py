import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure

from tqdm import tqdm
import numpy as np
from decouple import config

from utils.save_model import save_model
from dataset import CDANDataset
from models.cdan import CDAN


def train(model, optimizer, criterion, n_epoch, data_loaders: dict, device, save_dir_root, model_name):
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].to(device)
    perceptual_loss_weight = 0.5  # Adjust the weight as desired
    
    train_losses = np.zeros(n_epoch)
    val_losses = np.zeros(n_epoch)
    best_psnr = 0.0

    model.to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    since = time.time()

    for epoch in range(n_epoch):
        train_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        model.train()
        for inputs, targets in tqdm(data_loaders['train'], desc=f'Training... Epoch: {epoch + 1}/{EPOCHS}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            perceptual_loss = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))
            loss = loss + perceptual_loss
            train_loss += loss.item()
            train_psnr += psnr(outputs, targets)
            train_ssim += ssim(outputs, targets)

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(data_loaders['train'])
        train_psnr = train_psnr / len(data_loaders['train'])
        train_ssim = train_ssim / len(data_loaders['train'])

        with torch.no_grad():
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            model.eval()
            for inputs, targets in tqdm(data_loaders['validation'], desc=f'Validating... Epoch: {epoch + 1}/{EPOCHS}'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                perceptual_loss = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))
                loss = loss + perceptual_loss
                val_loss += loss.item()
                val_psnr += psnr(outputs, targets)
                val_ssim += ssim(outputs, targets)

            val_loss = val_loss / len(data_loaders['validation'])
            val_psnr = val_psnr / len(data_loaders['validation'])
            val_ssim = val_ssim / len(data_loaders['validation'])

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_model(model, optimizer, val_loss, val_psnr,
                           val_ssim, epoch, save_dir_root, model_name, device)

        # save epoch losses
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        print(f"Epoch [{epoch+1}/{n_epoch}]:")
        print(
            f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.4f}, Train SSIM: {train_ssim:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Validation PSNR: {val_psnr:.4f}, Validation SSIM: {val_ssim:.4f}")
        print('-'*20)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    INPUT_SIZE = config('INPUT_SIZE', default=200, cast=int)
    BATCH_SIZE = config('INPUT_SIZE', default=16, cast=int)
    EPOCHS = config('INPUT_SIZE', default=80, cast=int)
    LEARNING_RATE = config('INPUT_SIZE', default=1e-3, cast=float)

    # Configurations
    DATASET_DIR_ROOT = config('DATASET_DIR_ROOT')
    SAVE_DIR_ROOT = config('SAVE_DIR_ROOT')
    MODEL_NAME = config('MODEL_NAME')
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'

    train_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = CDANDataset(
        os.path.join(DATASET_DIR_ROOT, 'train', 'low'),
        os.path.join(DATASET_DIR_ROOT, 'train', 'high'),
        train_transforms,
        train_transforms
    )
    test_dataset = CDANDataset(
        os.path.join(DATASET_DIR_ROOT, 'test', 'low'),
        os.path.join(DATASET_DIR_ROOT, 'test', 'high'),
        test_transforms,
        test_transforms
    )

    data_loaders = {
        'train': DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        ),
        'validation': DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        )
    }

    model = CDAN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer, criterion, EPOCHS, data_loaders, DEVICE, SAVE_DIR_ROOT, MODEL_NAME)
