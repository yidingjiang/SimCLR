import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        h4 = F.relu(self.l_4(torch.cat((h3, noise[3]), 1)))
        norm = h4.norm(p=self.p, dim=(1, 2, 3), keepdim=True).detach()
        out = x + 0.05*96*96*3*h4.div(norm)
        return torch.clamp(out, 0., 1.) if self.clip else out
