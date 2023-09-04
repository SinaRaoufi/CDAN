import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


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


class CDAN(nn.Module):
    def __init__(self):
        super(CDAN, self).__init__()
        # Encoder
        self.encoder_conv1 = ConvBlock(3, 64)
        self.encoder_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_dp1 = nn.Dropout(0.2)

        self.encoder_conv2 = ConvBlock(64, 128)
        self.encoder_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_dp2 = nn.Dropout(0.2)

        self.encoder_conv3 = ConvBlock(128, 256)
        self.encoder_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_dp3 = nn.Dropout(0.2)

        self.encoder_conv4 = ConvBlock(256, 512)
        self.encoder_dp4 = nn.Dropout(0.2)
        
        # Bottleneck
        self.bottleneck = CBAM(512)

        # Decoder
        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.decoder_cbam1 = CBAM(256)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        
        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.decoder_cbam2 = CBAM(128)
        self.decoder_bn2 = nn.BatchNorm2d(128)
       
        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_cbam3 = CBAM(64)
        self.decoder_bn3 = nn.BatchNorm2d(64)
        
        self.decoder_conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(3)
        self.decoder_relu = nn.ReLU(inplace=True)

        # Dense blocks
        self.final_dense = DenseBlock(3, 3, 16, 4)
        self.dense1 = DenseBlock(64, 64, 16, 4)
        self.dense2 = DenseBlock(128, 128, 16, 4)
        self.dens3 = DenseBlock(256, 256, 16, 4)       
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.decoder_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        out = self.encoder_conv1(x)
        out = self.maxpool(out)
        dense1 = self.dense1(out)
        out = self.dp(out)
        skip_connections.append(out)

        out = self.encoder_conv2(out)
        out = self.maxpool(out)
        dense2 = self.dense2(out)
        out = self.dp(out)
        skip_connections.append(out)

        out = self.encoder_conv3(out)
        out = self.maxpool(out)
        dense3 = self.dens3(out)
        out = self.dp(out)
        skip_connections.append(out)

        out = self.encoder_conv4(out)
        out = self.dp(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        out = self.decoder_conv1(out)
        out = self.decoder_bn1(out)
        out = self.relu(out)
        out = torch.add(out, skip_connections[2])
        out = self.decoder_cbam1(out)

        out *= dense3
        out = self.decoder_conv2(out)
        out = self.decoder_bn2(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, skip_connections[1])
        out = self.decoder_cbam2(out)

        out *= dense2
        out = self.decoder_conv3(out)
        out = self.decoder_bn3(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, skip_connections[0])
        out = self.decoder_cbam3(out)

        out *= dense1
        out = self.decoder_conv4(out)
        out = self.decoder_bn4(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, x)

        dense_out = self.final_dense(out)
        out = self.decoder_sigmoid(dense_out)

        return out
