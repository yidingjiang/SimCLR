import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class IcmAugmentor(nn.Module):
    def __init__(self, num_mech, num_app=1, max_app_parallel=4, clip=True):
        super(IcmAugmentor, self).__init__()
        self.num_mech = num_mech
        self.num_app = num_app
        self.max_app_parallel = max_app_parallel
        self.clip = clip
        self.mechanisms = (Mechanism() for _ in range(self.num_mech))
        self.selected_mechanism = []

    def forward(self, x):
        out = x
        num_per_mech = x.size()[0] // self.max_app_parallel
        mech_label = []
        mech_idx = np.arange(self.num_mech)

        for _ in range(self.num_app):
            # sample which mechanisms to apply and create label
            chosen_idx = np.random.choice(mech_idx, self.max_app_parallel, True)
            chosen_mech = [self.mechanisms[i] for i in chosen_idx]
            mech_label.append(np.concatenate([[i] * num_per_mech for i in chosen_idx]))
            # split and apply
            split = torch.split(out, self.max_app_parallel)
            split = [m(s) for s, m in zip(split, chosen_mech)]
            # merge and clamp
            out = torch.cat(split)

        return out, mech_label


class Mechanism(nn.Module):

    def __init__(self, p=1, magnitude=0.05):
        super(Mechanism, self).__init__()
        self.p = p
        self.magnitude = magnitude
        self.transform = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2),
        )

    def forward(self, x):
        shape = x.size()
        h = self.transform(x)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
        out = x + self.magnitude * np.prod(shape[1:]) * h.div(norm).detach()
        return torch.clamp(out, 0., 1.) if self.clip else out


class Discriminator(nn.Module):

    def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
        super(Discriminator, self).__init__()
        self.num_mech = num_mech
        self.h_dim = 32 * (input_dim // (2**3))**2
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(self.h_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_mech)
        )

    def forward(self, x, x_t):
        x_cat = torch.cat([x, x_t], dim=1)
        return self.model(x_cat)
