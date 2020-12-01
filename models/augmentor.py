import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from style_transfer_model import ConvLayer
from style_transfer_model import ResidualBlock
from style_transfer_model import UpsampleConvLayer


class LpAugmentor(nn.Module):
    def __init__(self, p=1, noise_dim=3, clip=True):
        super(LpAugmentor, self).__init__()
        self.noise_dim = noise_dim
        self.p = p
        self.clip = clip

        self.l_1 = nn.Conv2d(self.noise_dim + 3, 64, 3, padding=1)
        self.l_2 = nn.Conv2d(self.noise_dim + 64, 64, 3, padding=1)
        self.l_3 = nn.Conv2d(self.noise_dim + 64, 64, 3, padding=1)
        self.l_4 = nn.Conv2d(self.noise_dim + 64, 3, 3, padding=1)

    def noise_shapes(self, input_dim):
        return [[3, input_dim, input_dim]] * 4

    def forward(self, x, noise):
        h1 = F.relu(self.l_1(torch.cat((x, noise[0]), 1)))
        h2 = F.relu(self.l_2(torch.cat((h1, noise[1]), 1)))
        h3 = F.relu(self.l_3(torch.cat((h2, noise[2]), 1)))
        h4 = self.l_4(torch.cat((h3, noise[3]), 1))
        norm = h4.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
        out = x + 0.05*96*96*3*h4.div(norm)
        return torch.clamp(out, 0., 1.) if self.clip else out


class LpAugmentorStyleTransfer(nn.Module):

    def __init__(self):
        super(LpAugmentorStyleTransfer, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        # self.res4 = ResidualBlock(128)
        # self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def noise_shapes(self, input_dim):
        return [[1, input_dim - 4, input_dim - 4]] * 3

    def forward(self, X, noise):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(torch.cat((y, noise[0]), 1))
        y = self.res2(torch.cat((y, noise[1]), 1))
        y = self.res3(torch.cat((y, noise[2]), 1))
        # y = self.res4(y)
        # y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
