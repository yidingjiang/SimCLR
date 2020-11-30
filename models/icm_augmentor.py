import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.distributions as tdist


class IcmAugmentor(nn.Module):
    def __init__(self, num_mech, num_app=1, max_app_parallel=4, clip=True, device="cuda"):
        super(IcmAugmentor, self).__init__()
        self.num_mech = num_mech
        self.num_app = num_app
        self.max_app_parallel = max_app_parallel
        self.clip = clip
        self.mechanisms = nn.ModuleList()
        for _ in range(num_mech):
          self.mechanisms.append(Mechanism())
        self.selected_mechanism = []
        self.device = device

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
            split = torch.split(out, num_per_mech)
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
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2),
        )

    def forward(self, x):
        shape = x.size()
        h = self.transform(x)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
        out = x + self.magnitude * np.prod(shape[1:]) * h.div(norm).detach()
        return torch.clamp(out, 0., 1.)


class Discriminator(nn.Module):

    def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
        super(Discriminator, self).__init__()
        self.num_mech = num_mech
        self.h_dim = 32 * (input_dim // (2**3))**2
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(128, 256)
        self.f2 = nn.Linear(256, num_mech)

    def forward(self, x):
        x_cat = torch.cat([x[0], x[1]], dim=1)
        out = self.model(x_cat)
        out = torch.mean(out, (2, 3))
        out = F.relu(self.f1(out))
        out = self.f2(out)
        return out

#============================== Noise version ==============================

class IcmAugmentorv2(nn.Module):
    def __init__(self, num_mech, num_app=1, max_app_parallel=4, clip=True, device="cuda"):
        super(IcmAugmentorv2, self).__init__()
        self.num_mech = num_mech
        self.num_app = num_app
        self.max_app_parallel = max_app_parallel
        self.clip = clip
        self.mechanisms = nn.ModuleList()
        for _ in range(num_mech):
          self.mechanisms.append(Mechanism())
        self.selected_mechanism = []
        self.device = device
        # self.normal_dist = tdist.Uniform(torch.tensor([-1.]), torch.tensor([1.]))

    def forward(self, x):
        out = x
        batch_size = x.size()[0]
        num_per_mech = batch_size // self.max_app_parallel
        mech_label = []
        mech_idx = np.arange(self.num_mech)
        for _ in range(self.num_app):
            # sample which mechanisms to apply and create label
            chosen_idx = np.random.choice(mech_idx, self.max_app_parallel, True)
            chosen_mech = [self.mechanisms[i] for i in chosen_idx]
            chosen_val = np.random.uniform(self.noise_bound[0], self.noise_bound[1], batch_size)
            mech_label.append(np.concatenate([[i] * num_per_mech for i in chosen_idx]))
            # split and apply
            split = torch.split(out, num_per_mech)
            # val_split = torch.split(torch.tensor(chosen_val).to(self.device), num_per_mech)
            val_split = np.split(chosen_val, self.max_app_parallel)
            split = [m(s, v) for s, m, v in zip(split, chosen_mech, val_split)]
            # merge and clamp
            out = torch.cat(split)
        return out, mech_label, chosen_val


class Mechanismv2(nn.Module):

    def __init__(self, p=1, magnitude=0.05):
        super(Mechanismv2, self).__init__()
        self.p = p
        self.magnitude = magnitude
        self.transform = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2),
        )

    def forward(self, x, v):
        s = x.size()
        v = torch.tensor(np.ones([s[0], 1, s[2], s[3]]) * v)
        x = torch.cat((x, v), 1)
        h = self.transform(x)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
        out = x + self.magnitude * np.prod(s[1:]) * h.div(norm).detach()
        return torch.clamp(out, 0., 1.)


class Discriminatorv2(nn.Module):

    def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
        super(Discriminatorv2, self).__init__()
        self.num_mech = num_mech
        self.h_dim = 32 * (input_dim // (2**3))**2
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(128, 256)
        self.f2 = nn.Linear(256, num_mech)
        self.v1 = nn.Linear(128, 32)
        self.v2 = nn.Linear(32, 1)

    def forward(self, x):
        x_cat = torch.cat([x[0], x[1]], dim=1)
        out = self.model(x_cat)
        out = torch.mean(out, (2, 3))  # Global avg
        id_out = F.relu(self.f1(out))
        id_pred = self.f2(id_out)
        val_out = F.relu(self.v1(out))
        val_pred = self.v2(id_out)
        return id_pred, val_pred
