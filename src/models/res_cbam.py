import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(x)
        out += residual
        out = self.relu(out)
        return out


class LLIE(nn.Module):
    def __init__(self):
        super(LLIE, self).__init__()
        self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(64)
        self.encoder_relu1 = nn.ReLU(inplace=True)
        self.encoder_residual1 = ResidualBlock(64)
        self.encoder_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(128)
        self.encoder_relu2 = nn.ReLU(inplace=True)
        self.encoder_residual2 = ResidualBlock(128)
        self.encoder_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.encoder_bn3 = nn.BatchNorm2d(256)
        self.encoder_relu3 = nn.ReLU(inplace=True)
        self.encoder_residual3 = ResidualBlock(256)
        self.encoder_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.encoder_bn4 = nn.BatchNorm2d(512)
        self.encoder_relu4 = nn.ReLU(inplace=True)
        self.encoder_residual4 = ResidualBlock(512)

        self.bottleneck = CBAM(512)

        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_relu1 = nn.ReLU(inplace=True)
        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(128)
        self.decoder_relu2 = nn.ReLU(inplace=True)
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(64)
        self.decoder_relu3 = nn.ReLU(inplace=True)
        self.decoder_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(3)
        self.decoder_sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []
        # Encoder
        out = self.encoder_conv1(x)
        out = self.encoder_bn1(out)
        out = self.encoder_relu1(out)
        out = self.encoder_residual1(out)
        out = self.encoder_maxpool1(out)
        skip_connections.append(out)
        out = self.encoder_conv2(out)
        out = self.encoder_bn2(out)
        out = self.encoder_relu2(out)
        out = self.encoder_residual2(out)
        out = self.encoder_maxpool2(out)
        skip_connections.append(out)
        out = self.encoder_conv3(out)
        out = self.encoder_bn3(out)
        out = self.encoder_relu3(out)
        out = self.encoder_residual3(out)
        out = self.encoder_maxpool3(out)
        skip_connections.append(out)
        out = self.encoder_conv4(out)
        out = self.encoder_bn4(out)
        out = self.encoder_relu4(out)
        out = self.encoder_residual4(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        out = self.decoder_conv1(out)
        out = self.decoder_bn1(out)
        out = self.decoder_relu1(out)
        out = torch.add(out, skip_connections[2])
        out = self.decoder_conv2(out)
        out = self.decoder_bn2(out)
        out = self.decoder_relu2(out)
        out = torch.add(out, skip_connections[1])
        out = self.decoder_conv3(out)
        out = self.decoder_bn3(out)
        out = self.decoder_relu3(out)
        out = torch.add(out, skip_connections[0])
        out = self.decoder_conv4(out)
        out = self.decoder_bn4(out)
        out = self.decoder_sigmoid(out)

        return out




