import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.distributions as tdist

from models.style_transfer_model import TransformerNet
from models.style_transfer_model import ConvLayer
from models.style_transfer_model import ResidualBlock
from models.style_transfer_model import UpsampleConvLayer


class IcmAugmentor(nn.Module):
    def __init__(
        self,
        num_mech,
        augmentor_type="cnn",
        num_app=1,
        max_app_parallel=4,
        clip=True,
        device="cuda",
    ):
        super(IcmAugmentor, self).__init__()
        self.num_mech = num_mech
        self.num_app = num_app
        self.max_app_parallel = max_app_parallel
        self.clip = clip
        self.mechanisms = nn.ModuleList()

        self.augmentor_type = augmentor_type
        if self.augmentor_type == "cnn":
            self.mechanism_obj = Mechanism
        elif self.augmentor_type == "style_transfer":
            self.mechanism_obj = ResNetMechanism
        else:
            raise ValueError("Unrecognized mechanism type: {}".format(self.augmentor_type))

        for _ in range(num_mech):
            self.mechanisms.append(self.mechanism_obj())
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
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True).detach()
        out = x + self.magnitude * np.prod(shape[1:]) * h.div(norm)
        return torch.clamp(out, 0.0, 1.0)


class ResNetMechanism(nn.Module):
    def __init__(self, p=1, magnitude=0.05):
        super(ResNetMechanism, self).__init__()
        self.p = p
        self.magnitude = magnitude
        self.transform = TransformerNet()

    def forward(self, x):
        shape = x.size()
        h = self.transform(x)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True).detach()
        out = x + self.magnitude * np.prod(shape[1:]) * h.div(norm)
        return torch.clamp(out, 0.0, 1.0)


class Discriminator(nn.Module):
    def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
        super(Discriminator, self).__init__()
        self.num_mech = num_mech
        self.h_dim = 32 * (input_dim // (2 ** 3)) ** 2
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


# ============================== Noise version ==============================


class IcmAugmentorv2(nn.Module):

    def __init__(
        self, num_mech, num_app=1, max_app_parallel=4, clip=True, device="cuda"
    ):
        super(IcmAugmentorv2, self).__init__()
        self.num_mech = num_mech
        self.num_app = num_app
        self.max_app_parallel = max_app_parallel
        self.clip = clip
        self.mechanisms = nn.ModuleList()
        for _ in range(num_mech):
            self.mechanisms.append(Mechanismv2(device=device))
        self.selected_mechanism = []
        self.device = device
        self.noise_bound = [-1, 1]
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
            chosen_val = np.random.uniform(
                self.noise_bound[0], self.noise_bound[1], batch_size
            )
            mech_label.append(np.concatenate([[i] * num_per_mech for i in chosen_idx]))
            # split and apply
            split = torch.split(out, num_per_mech)
            # val_split = torch.split(torch.tensor(chosen_val).to(self.device), num_per_mech)
            val_split = np.split(chosen_val, self.max_app_parallel)
            split = [m((s, v)) for s, m, v in zip(split, chosen_mech, val_split)]
            # merge and clamp
            out = torch.cat(split)
        return out, {"id": mech_label, "value": chosen_val}


class Mechanismv2(nn.Module):

    def __init__(self, device="cuda", p=1, magnitude=0.05):
        super(Mechanismv2, self).__init__()
        self.p = p
        self.magnitude = magnitude
        self.device = device
        self.transform = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2),
        )

    def forward(self, x):
        x, v = x[0], x[1]
        s = x.size()
        v = torch.tensor(
            np.ones([s[0], 1, s[2], s[3]]) * np.expand_dims(v, [1, 2, 3]),
            dtype=torch.float,
        ).to(self.device)
        x_cat = torch.cat((x, v), 1)
        h = self.transform(x_cat)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
        # print("x", x.size(), "v", v.size(), "norm", norm.size(), "h", h.size())
        out = x + self.magnitude * np.prod(s[1:]) * h.div(norm).detach()
        return torch.clamp(out, 0.0, 1.0)


class Discriminatorv2(nn.Module):

    def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
        super(Discriminatorv2, self).__init__()
        self.num_mech = num_mech
        self.h_dim = 32 * (input_dim // (2 ** 3)) ** 2
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
        val_pred = self.v2(val_out)
        return id_pred, val_pred


# ============================== InfoGAN version ==============================


class IcmAugmentorv3(nn.Module):
    def __init__(
        self, num_mech, augmentor_type="cnn", num_app=1, max_app_parallel=4, clip=True, device="cuda"
    ):
        super(IcmAugmentorv3, self).__init__()
        self.num_mech = num_mech
        self.num_app = num_app
        self.max_app_parallel = max_app_parallel
        self.clip = clip

        self.augmentor_type = augmentor_type
        if self.augmentor_type == "cnn":
            self.mechanism_obj = Mechanismv3
        elif self.augmentor_type == "style_transfer":
            self.mechanism_obj = ResNetMechanismv3
        else:
            raise ValueError("Unrecognized mechanism type: {}".format(self.augmentor_type))

        self.mechanism = self.mechanism_obj(self.num_mech, device=device)
        self.selected_mechanism = []
        self.device = device
        self.noise_bound = [-1, 1]

    def forward(self, x):
        out = x
        batch_size = x.size()[0]
        chosen_val = np.random.uniform(
            self.noise_bound[0], self.noise_bound[1], [batch_size, self.num_mech]
        )
        chosen_val = np.float32(chosen_val)
        out = self.mechanism((x, chosen_val))
        return out, {"value": chosen_val}


class Mechanismv3(nn.Module):
    def __init__(self, num_mech, device="cuda", p=1, magnitude=0.05):
        super(Mechanismv3, self).__init__()
        self.p = p
        self.magnitude = magnitude
        self.device = device
        self.num_mech = num_mech

        self.down_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
        )
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2),
        )
        self.f1 = nn.Linear(self.num_mech, 128)
        self.f2 = nn.Linear(128, 6 * 6 * 8)
        self.code_up = nn.Sequential(
            nn.ConvTranspose2d(8, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x, v = x[0], x[1]
        s = x.size()
        v = torch.tensor(v).to(self.device)
        v = F.relu(self.f1(v))
        v = F.relu(self.f2(v))
        v = torch.reshape(v, (s[0], 8, 6, 6))
        v = F.sigmoid(self.code_up(v))
        x_down = self.down_conv(x)
        x_combined = torch.cat([x_down,v], dim=1)
        h = self.up_conv(x_combined)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True)
        out = x + self.magnitude * np.prod(s[1:]) * h.div(norm).detach()
        return torch.clamp(out, 0.0, 1.0)


class ResNetMechanismv3(nn.Module):
    def __init__(self, num_mech, device="cuda", p=1, magnitude=0.05):
        super(ResNetMechanismv3, self).__init__()
        self.p = p
        self.num_mech = num_mech
        self.magnitude = magnitude
        self.device = device
        self.transform = TransformerNet()

        self.f1 = nn.Linear(self.num_mech, 3*8*8)
        self.code_up = nn.Sequential(
            nn.ConvTranspose2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1),
        )
        self.upsample = torch.nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, x):
        x, v = x[0], x[1]
        v = torch.tensor(v).to(self.device)
        shape = x.size()

        noise = self.f1(v)
        noise = torch.reshape(noise, (shape[0], 3, 8, 8))
        noise = self.code_up(noise)
        # print(noise.size())
        noise = self.upsample(noise)

        combined = x * torch.sigmoid(noise)
        h = self.transform(combined)
        norm = h.norm(p=self.p, dim=(1, 2, 3), keepdim=True).detach()
        out = x + self.magnitude * np.prod(shape[1:]) * h.div(norm)
        return torch.clamp(out, 0.0, 1.0)


# class Discriminatorv3(nn.Module):
#     def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
#         super(Discriminatorv3, self).__init__()
#         self.num_mech = num_mech
#         self.h_dim = 32 * (input_dim // (2 ** 3)) ** 2
#         self.model = nn.Sequential(
#             nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         self.flatten = nn.Flatten()
#         self.v1 = nn.Linear(128, 128)
#         self.v2 = nn.Linear(128, self.num_mech)

#     def forward(self, x):
#         x_cat = torch.cat([x[0], x[1]], dim=1)
#         out = self.model(x_cat)
#         out = torch.mean(out, (2, 3))  # Global avg
#         val_out = F.relu(self.v1(out))
#         val_pred = self.v2(val_out)
#         return val_pred


class Discriminatorv3(nn.Module):
    def __init__(self, num_mech, input_dim=32, p=1, magnitude=0.05):
        super(Discriminatorv3, self).__init__()
        self.num_mech = num_mech
        self.h_dim = 32 * (input_dim // (2 ** 3)) ** 2
        self.original_preprocess = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.v1 = nn.Linear(256, 256)
        self.v2 = nn.Linear(256, self.num_mech)

    def forward(self, x):
        # second one is the original
        x_o_proj = self.original_preprocess(x[1])
        x_cat = torch.cat([x[0], x_o_proj], dim=1)
        out = self.model(x_cat)
        out = torch.mean(out, (2, 3))  # Global avg
        val_out = F.relu(self.v1(out))
        val_pred = self.v2(val_out)
        return val_pred
