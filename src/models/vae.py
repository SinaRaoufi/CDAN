import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM
from models.res_bam import BAM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(channels)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn(out)
        out += residual
        # out = self.cbam(out)
        out = self.relu(out)
        return out


class MixVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(MixVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv1 = ConvBlock(3, 64)
        self.encoder_residual1 = ResidualBlock(64)
        self.encoder_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = ConvBlock(64, 128)
        self.encoder_residual2 = ResidualBlock(128)
        self.encoder_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = ConvBlock(128, 256)
        self.encoder_residual3 = ResidualBlock(256)
        self.encoder_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = ConvBlock(256, 512)
        self.encoder_residual4 = ResidualBlock(512)

        self.bottleneck_fc1 = nn.Linear(512 * 4 * 4, latent_dim)
        self.bottleneck_fc2 = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.decoder_bam1 = CBAM(256)
        self.decoder_res1 = ResidualBlock(256)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder_bam2 = CBAM(128)
        self.decoder_res2 = ResidualBlock(128)
        self.decoder_bn2 = nn.BatchNorm2d(128)
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_bam3 = CBAM(64)
        self.decoder_res3 = ResidualBlock(64)
        self.decoder_bn3 = nn.BatchNorm2d(64)
        self.decoder_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.decoder_bam4 = CBAM(3)
        self.decoder_res4 = ResidualBlock(3)
        self.decoder_bn4 = nn.BatchNorm2d(3)
        self.decoder_relu = nn.ReLU(inplace=True)
        self.decoder_sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.encoder_conv1(x)
        out = self.encoder_residual1(out)
        out = self.encoder_maxpool1(out)
        out = self.encoder_conv2(out)
        out = self.encoder_residual2(out)
        out = self.encoder_maxpool2(out)
        out = self.encoder_conv3(out)
        out = self.encoder_residual3(out)
        out = self.encoder_maxpool3(out)
        out = self.encoder_conv4(out)
        out = self.encoder_residual4(out)
        out = out.view(out.size(0), -1)
        mean = self.bottleneck_fc1(out)
        logvar = self.bottleneck_fc2(out)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        out = self.decoder_fc(z)
        out = out.view(out.size(0), 512, 4, 4)
        out = self.decoder_conv1(out)
        out = self.decoder_bn1(out)
        out = self.decoder_relu(out)
        out = self.decoder_conv2(out)
        out = self.decoder_bn2(out)
        out = self.decoder_relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.decoder_conv3(out)
        out = self.decoder_bn3(out)
        out = self.decoder_relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.decoder_res3(out)
        out = self.decoder_conv4(out)
        out = self.decoder_bn4(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.decoder_sigmoid(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar
