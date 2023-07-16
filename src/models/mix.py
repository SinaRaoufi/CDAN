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
    
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = ConvBlock(in_channels, out_channels, kernel_size=5, stride=1, padding=1)

#     def forward(self, x):
#         stream1 = self.conv(x)
#         stream2 = self.conv(x)
#         multiply = stream1 * stream2

        
#         return out


# class DenseBlock(nn.Module):
#     def __init__(self, in_channels, growth_rate, num_layers):
#         super(DenseBlock, self).__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(self._make_dense_layer(in_channels, growth_rate))
#             in_channels += growth_rate

#     def forward(self, x):
#         features = [x]
#         for layer in self.layers:
#             out = layer(torch.cat(features, dim=1))
#             features.append(out)
#         return torch.cat(features, dim=1)

#     def _make_dense_layer(self, in_channels, growth_rate):
#         return nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
#         )

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels, growth_rate))
            in_channels += growth_rate

        self.transition_layer = self._make_transition_layer(in_channels, out_channels)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)

        out = self.transition_layer(torch.cat(features, dim=1))
        return out

    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
        )

    def _make_transition_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        # out = self.dropout(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv(out)
        # out = self.dropout(x)
        out = self.bn(out)
        out += residual
        # out = self.cbam(out)
        out = self.relu(out)
        return out


# class Mix(nn.Module):
#     def __init__(self):
#         super(Mix, self).__init__()
#         self.encoder_conv1 = ConvBlock(3, 64)
#         self.encoder_residual1 = ResidualBlock(64)
#         self.encoder_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder_conv2 = ConvBlock(64, 128)
#         self.encoder_residual2 = ResidualBlock(128)
#         self.encoder_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder_conv3 = ConvBlock(128, 256)
#         self.encoder_residual3 = ResidualBlock(256)
#         self.encoder_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder_conv4 = ConvBlock(256, 512)
#         self.encoder_residual4 = ResidualBlock(512)

#         self.bottleneck = CBAM(512)

#         self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
#         self.decoder_bam1 = CBAM(256)
#         self.decoder_res1 = ResidualBlock(256)
#         self.decoder_bn1 = nn.BatchNorm2d(256)
#         self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
#         self.decoder_bam2 = CBAM(128)
#         self.decoder_res2 = ResidualBlock(128)
#         self.decoder_bn2 = nn.BatchNorm2d(128)
#         self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.decoder_bam3 = CBAM(64)
#         self.decoder_res3 = ResidualBlock(64)
#         self.decoder_bn3 = nn.BatchNorm2d(64)
#         self.decoder_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
#         self.decoder_bam4 = CBAM(3)
#         self.decoder_res4 = ResidualBlock(3)
#         self.decoder_bn4 = nn.BatchNorm2d(3)
#         self.decoder_relu = nn.ReLU(inplace=True)
#         self.decoder_sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         skip_connections = []
#         # Encoder
#         out = self.encoder_conv1(x)
#         out = self.encoder_residual1(out)
#         out = self.encoder_maxpool1(out)
#         skip_connections.append(out)
#         out = self.encoder_conv2(out)
#         out = self.encoder_residual2(out)
#         out = self.encoder_maxpool2(out)
#         skip_connections.append(out)
#         out = self.encoder_conv3(out)
#         out = self.encoder_residual3(out)
#         out = self.encoder_maxpool3(out)
#         skip_connections.append(out)
#         out = self.encoder_conv4(out)
#         out = self.encoder_residual4(out)

#         # Bottleneck
#         out = self.bottleneck(out)

#         # Decoder
#         out = self.decoder_conv1(out)
#         out = self.decoder_bn1(out)
#         out = self.decoder_relu(out)
#         out = torch.add(out, skip_connections[2])
#         out = self.decoder_res1(out)
#         # out = self.decoder_bam1(out)
#         out = self.decoder_conv2(out)
#         out = self.decoder_bn2(out)
#         out = self.decoder_relu(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear',  align_corners=False)
#         out = torch.add(out, skip_connections[1])
#         out = self.decoder_res2(out)
#         # out = self.decoder_bam2(out)
#         out = self.decoder_conv3(out)
#         out = self.decoder_bn3(out)
#         out = self.decoder_relu(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear',  align_corners=False)
#         out = self.decoder_res3(out)
#         out = torch.add(out, skip_connections[0])
#         # out = self.decoder_bam3(out)
#         out = self.decoder_conv4(out)
#         out = self.decoder_bn4(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear',  align_corners=False)
#         # out = self.decoder_res4(out)
#         # out = self.decoder_bam4(out)
#         out = self.decoder_sigmoid(out)
#         return out


class Mix(nn.Module):
    def __init__(self):
        super(Mix, self).__init__()
        self.encoder_conv1 = ConvBlock(3, 64)
        self.encoder_residual1 = ResidualBlock(64)
        self.encoder_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_bam1 = BAM(64)

        self.encoder_conv2 = ConvBlock(64, 128)
        self.encoder_residual2 = ResidualBlock(128)
        self.encoder_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_bam2 = BAM(128)

        self.encoder_conv3 = ConvBlock(128, 256)
        self.encoder_residual3 = ResidualBlock(256)
        self.encoder_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_bam3 = BAM(256)

        self.encoder_conv4 = ConvBlock(256, 512)
        self.encoder_residual4 = ResidualBlock(512)

        self.dense1 = DenseBlock(64, 64, 4, 4)
        self.dense2 = DenseBlock(128, 128, 4, 4)

        self.bottleneck = CBAM(512)

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

        self.dense_block = DenseBlock(3, 3, 3, 4)  # Example configuration: 4 layers with growth rate of 32
        self.final_conv = ConvBlock(6, 3)

    def forward(self, x):
        skip_connections = []
        # Encoder

        out = self.encoder_conv1(x)
        bam1 = self.encoder_bam1(out)
        bam1 = self.encoder_maxpool1(bam1)

        out = self.encoder_residual1(out)
        out = self.encoder_maxpool1(out)
        
        out = torch.add(out, bam1)
        skip_connections.append(out)

        # dense_block_1 = self.dense1(out) # torch.Size([16, 256, 128, 128])

        out = self.encoder_conv2(out)
        bam2 = self.encoder_bam2(out)
        bam2 = self.encoder_maxpool2(bam2)

        out = self.encoder_residual2(out)
        out = self.encoder_maxpool2(out)
        # bam2 = self.encoder_bam2(bam1)
        out = torch.add(out, bam2)
        skip_connections.append(out)

        # dense_block_2 = self.dense2(out) # torch.Size([16, 256, 128, 128])
        
        out = self.encoder_conv3(out)
        bam3 = self.encoder_bam3(out)
        bam3 = self.encoder_maxpool3(bam3)

        out = self.encoder_residual3(out)
        out = self.encoder_maxpool3(out)
        out = torch.add(out, bam3)
        skip_connections.append(out)

        out = self.encoder_conv4(out)
        out = self.encoder_residual4(out)
        
        # Bottleneck
        out = self.bottleneck(out)

        # Decoder

        out = self.decoder_conv1(out)
        out = self.decoder_bn1(out)
        # out = self.decoder_relu(out)
        out = self.decoder_res1(out)
        out = torch.add(out, skip_connections[2])
        # out = self.decoder_bam1(out)

        out = self.decoder_conv2(out)
        out = self.decoder_bn2(out)
        # out = self.decoder_relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.decoder_res2(out)
        out = torch.add(out, skip_connections[1])
        # out = self.decoder_bam2(out)
        # out = torch.add(out, dense_block_2)

        out = self.decoder_conv3(out)
        out = self.decoder_bn3(out)
        # out = self.decoder_relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.decoder_res3(out)
        out = torch.add(out, skip_connections[0])
        # out = self.decoder_bam3(out)
        # out = torch.add(out, dense_block_1)

        out = self.decoder_conv4(out)
        out = self.decoder_bn4(out)
        out = self.decoder_res4(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        # out = self.decoder_bam4(out)
        # out = self.decoder_sigmoid(out)

        # Dense Block
        
        # dense_out = self.decoder_sigmoid(dense_out)
        # Concatenate Dense Block output with decoder output
        # out = torch.cat([out, dense_out], dim=1)
        # print(out.shape)
        # out += dense_out
        # out = self.decoder_bn4(out)
        # out = self.final_conv(out)
        out = self.decoder_sigmoid(out)
        
        return out
