import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import numpy as np

from utils.save_model import save_model
from .dataset import LLIDataset
from .model import AutoEncoder


def train(model, optimizer, criterion, n_epoch,
          data_loaders: dict, device, lr_scheduler=None
          ):
    train_losses = np.zeros(n_epoch)
    val_losses = np.zeros(n_epoch)

    model.to(device)

    since = time.time()

    for epoch in range(n_epoch):
        train_loss = 0.0
        model.train()
        for inputs, targets in tqdm(data_loaders['train'], desc=f'Training... Epoch: {epoch + 1}/{EPOCHS}'):

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            # scheduler.step()

        train_loss = train_loss / len(data_loaders['train'].dataset)

        with torch.no_grad():
            val_loss = 0.0
            model.eval()
            for inputs, targets in tqdm(data_loaders['validation'], desc=f'Validating... Epoch: {epoch + 1}/{EPOCHS}'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            val_loss = val_loss / len(data_loaders['validation'].dataset)


        # save epoch losses
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        print(f"Epoch {epoch+1}/{n_epoch}:")
        print(f"Train Loss: {train_loss:.2f}")
        print(f"Validation Loss: {val_loss:.2f}")
        print('-'*20)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    save_model(model, 'new_model.pt')


if __name__ == '__main__':
    INPUT_SIZE = 224
    # DATASET_DIR_ROOT = "/home/novin/Desktop/aptos"
    BATCH_SIZE = 32
    EPOCHS = 80

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"

    train_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = LLIDataset()
    validation_dataset = LLIDataset()
    test_dataset = LLIDataset()

    data_loaders = {
        "train": DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5
        ),
        "validation": DataLoader(
            validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5
        )
    }

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, criterion, EPOCHS, data_loaders, device)