import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
import numpy as np
from decouple import config

from dataset import CDANDataset
from models.cdan import CDAN

import optuna


def objective(trial):
    # Search Space
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    epochs = trial.suggest_int('epochs', 50, 180, 5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    perceptual_loss_weight = trial.suggest_float('perceptual_loss_weight', 0.15 , 1)
    vgg_layers = trial.suggest_categorical('vgg_layers', [16, 18, 20, 23, 25])

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
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        ),
        'validation': DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
    }

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'

    model = CDAN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    vgg = models.vgg19(
        weights=models.VGG19_Weights.IMAGENET1K_V1).features[:vgg_layers].to(device)
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    best_psnr = 0.0

    for epoch in range(epochs):
        train_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        model.train()
        for inputs, targets in tqdm(data_loaders['train'], desc=f'Training... Epoch: {epoch + 1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            perceptual_loss = perceptual_loss_weight * \
                F.mse_loss(vgg(outputs), vgg(targets))
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
            val_psnr = 0.0
            val_loss = 0.0
            val_ssim = 0.0
            model.eval()
            for inputs, targets in tqdm(data_loaders['validation'], desc=f'Validating... Epoch: {epoch + 1}/{epochs}'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                perceptual_loss = perceptual_loss_weight * \
                    F.mse_loss(vgg(outputs), vgg(targets))
                loss = loss + perceptual_loss
                val_loss += loss.item()
                val_psnr += psnr(outputs, targets)
                val_ssim += ssim(outputs, targets)


            val_loss = val_loss / len(data_loaders['validation'])
            val_psnr = val_psnr / len(data_loaders['validation'])
            val_ssim = val_ssim / len(data_loaders['validation'])

            if val_psnr > best_psnr:
                best_psnr = val_psnr

        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        print(f"Epoch [{epoch+1}/{epochs}]:")
        print(
            f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.4f}, Train SSIM: {train_ssim:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Validation PSNR: {val_psnr:.4f}, Validation SSIM: {val_ssim:.4f}")
        print('-'*20)

    return best_psnr


if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    INPUT_SIZE = 200
    DATASET_DIR_ROOT = config('DATASET_DIR_ROOT')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_psnr = study.best_value

    print('Best hyperparameters found:')
    for param, value in best_params.items():
        print(f'{param}: {value}')
    print(f'Best PSNR value: {best_psnr:.4f}')
