import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out += residual
#         out = self.relu(out)
#         return out


# class LLIE(nn.Module):
#     def __init__(self):
#         super(LLIE, self).__init__()

#         # Encoder
#         self.encoder_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.encoder_relu1 = nn.ReLU(inplace=True)
#         self.encoder_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder_residual1 = ResidualBlock(64)
#         self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.encoder_relu2 = nn.ReLU(inplace=True)
#         self.encoder_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder_residual2 = ResidualBlock(128)
#         self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.encoder_relu3 = nn.ReLU(inplace=True)
#         self.encoder_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder_residual3 = ResidualBlock(256)
#         self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
#         self.encoder_relu4 = nn.ReLU(inplace=True)

#         # Bottleneck
#         self.bottleneck = CBAM(512)
# # Decoder
#         self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.decoder_relu1 = nn.ReLU(inplace=True)
#         self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.decoder_relu2 = nn.ReLU(inplace=True)
#         self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.decoder_relu3 = nn.ReLU(inplace=True)
#         self.decoder_residual = ResidualBlock(64)
#         self.decoder_conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
#         self.decoder_sigmoid = nn.Sigmoid()

#         # Skip connections
#         self.skip_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
#         self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
#         self.skip_conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # Encoder
#         skip_connections = []
#         out = self.encoder_conv1(x)
#         out = self.encoder_relu1(out)
#         out = self.encoder_maxpool1(out)
#         out = self.encoder_residual1(out)
#         skip_connections.append(self.skip_conv1(x))  # Skip connection 1
#         out = self.encoder_conv2(out)
#         out = self.encoder_relu2(out)
#         out = self.encoder_maxpool2(out)
#         out = self.encoder_residual2(out)
#         skip_connections.append(self.skip_conv2(out))  # Skip connection 2
#         out = self.encoder_conv3(out)
#         out = self.encoder_relu3(out)
#         out = self.encoder_maxpool3(out)
#         out = self.encoder_residual3(out)
#         skip_connections.append(self.skip_conv3(out))  # Skip connection 3
#         out = self.encoder_conv4(out)
#         out = self.encoder_relu4(out)

#         # Bottleneck
#         out = self.bottleneck(out)

#         # Decoder
#         out = self.decoder_conv1(out)
#         out = self.decoder_relu1(out)
#         skip_3_transposed = F.interpolate(skip_connections[2], scale_factor=8, mode='nearest')
#         out = torch.cat([out, skip_3_transposed], dim=1)  # Concatenate skip connection 3
#         out = self.decoder_conv2(out)
#         out = self.decoder_relu2(out)
#         skip_2_transposed = F.interpolate(skip_connections[1], scale_factor=4, mode='nearest')
#         out = torch.cat([out, skip_2_transposed], dim=1)  # Concatenate skip connection 2
#         out = self.decoder_conv3(out)
#         out = self.decoder_relu3(out)
#         skip_1_transposed = F.interpolate(skip_connections[0], scale_factor=2, mode='nearest')
#         out = torch.cat([out, skip_1_transposed], dim=1)  # Concatenate skip connection 1
#         out = self.decoder_residual(out)
#         out = self.decoder_conv4(out)
#         out = self.decoder_sigmoid(out)
#         return out
    




class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)         
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)         
        self.relu = nn.ReLU()     
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)      
        self.b = conv_block(512, 1024)      
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)     
    
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)        
        b = self.b(p4)      
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)         
        outputs = self.outputs(d4)        
        return outputs



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


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


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
        # print(f'encoder: {out.shape}')
        out = self.encoder_conv2(out)
        out = self.encoder_bn2(out)
        out = self.encoder_relu2(out)
        out = self.encoder_residual2(out)
        out = self.encoder_maxpool2(out)
        skip_connections.append(out)
        # print(f'encoder: {out.shape}')
        out = self.encoder_conv3(out)
        out = self.encoder_bn3(out)
        out = self.encoder_relu3(out)
        out = self.encoder_residual3(out)
        out = self.encoder_maxpool3(out)
        skip_connections.append(out)
        # print(f'encoder: {out.shape}')
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




