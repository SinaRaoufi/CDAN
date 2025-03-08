import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.models as models

from tqdm import tqdm
from PIL import Image
import numpy as np

from models.base import BaseModel
from utils.post_processing import enhance_color, enhance_contrast


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        """Must to init BaseModel with kwargs."""
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def composite_loss(self, outputs, targets):
        """Computes the composite loss by combining L2 loss and perceptual (VGG) loss."""
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:20].to(self.device)
        perceptual_loss_weight = 0.25
        loss = self.criterion(outputs, targets)
        perceptual_loss = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))

        return loss + perceptual_loss
    
    def generate_output_images(self, outputs, filenames, save_dir):
        """Generates and saves output images to the specified directory."""
        os.makedirs(save_dir, exist_ok=True)
        for i, output_image in enumerate(outputs):
            output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)

            output_path = os.path.join(save_dir, filenames[i])
            output_image.save(output_path)
        print(f'{len(outputs)} output images generated and saved to {save_dir}')


    def train_step(self):
        """Trains the model."""
        train_losses = np.zeros(self.epoch)
        best_loss = float('inf')
        self.network.to(self.device)

        for epoch in range(self.epoch):
            train_loss = 0.0
            dataloader_iter = tqdm(
                self.dataloader, desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.composite_loss(outputs, targets)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            train_loss = train_loss / len(self.dataloader)

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model(self.network)

            train_losses[epoch] = train_loss

            print(f"Epoch [{epoch + 1}/{self.epoch}] Train Loss: {train_loss:.4f}")


    def test_step(self):
      """Test the model."""
      path = os.path.join(self.model_path, self.model_name)
      self.network.load_state_dict(torch.load(path))
      self.network.eval()

      psnr = PeakSignalNoiseRatio().to(self.device)
      ssim = StructuralSimilarityIndexMeasure().to(self.device)
      lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

      all_outputs = []  # Accumulate outputs from all batches here
      all_filenames = []

      with torch.no_grad():
          test_loss = 0.0
          test_psnr = 0.0
          test_ssim = 0.0
          test_lpips = 0.0
          self.network.eval()
          self.optimizer.zero_grad()
          
          if self.is_dataset_paired:
              for inputs, targets in tqdm(self.dataloader, desc='Testing...'):
                  inputs, targets = inputs.to(self.device), targets.to(self.device)
                  outputs = self.network(inputs)
                  if self.apply_post_processing:
                      outputs = enhance_contrast(outputs, contrast_factor=1.12)
                      outputs = enhance_color(outputs, saturation_factor=1.35)
                  loss = self.criterion(outputs, targets)
                  test_loss += loss.item()
                  test_psnr += psnr(outputs, targets)
                  test_ssim += ssim(outputs, targets)
                  test_lpips += lpips(outputs, targets)
                  all_outputs.append(outputs)  # Append outputs of current batch
          else:
              for inputs,filenames in tqdm(self.dataloader, desc='Testing...'):
                  inputs = inputs.to(self.device)
                  outputs = self.network(inputs)
                  all_outputs.append(outputs)  # Append outputs of current batch
                  all_filenames.extend(filenames)

          if self.is_dataset_paired:
              test_loss = test_loss / len(self.dataloader)
              test_psnr = test_psnr / len(self.dataloader)
              test_ssim = test_ssim / len(self.dataloader)
              test_lpips = test_lpips / len(self.dataloader)
              print(
                  f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}, Test LPIPS: {test_lpips:.4f}'
              )

          # Concatenate outputs from all batches into one tensor
          all_outputs = torch.cat(all_outputs, dim=0)
          
          # Generate output images from the concatenated tensor
          self.generate_output_images(all_outputs, all_filenames, self.output_images_path)
