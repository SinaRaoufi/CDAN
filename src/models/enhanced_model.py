import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .cbam import CBAM


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EnhancedAutoEncoder(nn.Module):
    def __init__(self):
        super(EnhancedAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )
        self.bottleneck = CBAM(512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        equalized_img = transforms.functional.equalize(x.to(torch.uint8))
        loged_img = torch.log(x + 1e-8)
        gamma_img = transforms.functional.adjust_gamma(x, gamma=1.5)


        log_encoded = self.encoder(loged_img)
        log_encoded = self.bottleneck(log_encoded)
        log_decoded = self.decoder(log_encoded)

        equalized_encoded = self.encoder(equalized_img.to(torch.float32))
        equalized_encoded = self.bottleneck(equalized_encoded)
        equalized_decoded = self.decoder(equalized_encoded)

        gamma_encoded = self.encoder(gamma_img)
        gamma_encoded = self.bottleneck(gamma_encoded)
        gamma_decoded = self.decoder(gamma_encoded)

        concatenated = torch.add(log_decoded, equalized_decoded)
        concatenated = torch.add(gamma_decoded, concatenated)
        concatenated = torch.add(x, concatenated)
        return concatenated
