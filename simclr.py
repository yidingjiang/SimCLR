import torch
from models.resnet_simclr import ResNetSimCLR
from models.augmentor import LpAugmentor
from models.augmentor import LpAugmentorSpecNorm
from models.augmentor import LpAugmentorStyleTransfer
from models.augmentor import LpAugmentorTransformer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributions as tdist
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append("./apex")
    from apex import amp

    apex_support = True
except:
    print(
        "Please install apex for mixed precision training from: https://github.com/NVIDIA/apex"
    )
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            "./config.yaml", os.path.join(model_checkpoints_folder, "config.yaml")
        )


class SimCLRAdv(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config["batch_size"], **config["loss"]
        )
        self.normal_dist = tdist.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        self.augmentor_type = (
            config["augmentor_type"] if "augmentor_type" in config else "cnn"
        )
        self.augmentor_loss_type = (
            config["augmentor_loss_type"]
            if "augmentor_loss_type" in config
            else "linear"
        )

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "device" in self.config:
            device = self.config["device"]
        print("Running on:", device)
        return device

    def _adv_step(self, model, augmentor, xis, xjs, n_iter):
        shape = augmentor.noise_shapes(eval(self.config["dataset"]["input_shape"])[0])
        noise = [
            torch.squeeze(
                self.normal_dist.sample([self.config["batch_size"]] + s), -1
            ).to(self.device)
            for s in shape
        ]
        xis = augmentor(xis, noise)
        noise = [
            torch.squeeze(
                self.normal_dist.sample([self.config["batch_size"]] + s), -1
            ).to(self.device)
            for s in shape
        ]
        xjs = augmentor(xjs, noise)

        ris, zis = model(xis)  # [N,C]
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        if self.augmentor_type == "cnn":
            if self.config["normalization_type"] == "original":
                augmentor = LpAugmentor(clip=self.config["augmentor_clip_output"])
                augmentor.to(self.device)
            elif self.config["normalization_type"] == "spectral":
                augmentor = LpAugmentorSpecNorm(clip=self.config["augmentor_clip_output"])
                augmentor.to(self.device)
            else:
                raise ValueError(
                    "Unregonized normalization type: {}".format(
                        self.config["normalization_type"]
                    )
                )
        elif self.augmentor_type == "style_transfer":
            augmentor = LpAugmentorStyleTransfer(clip=self.config["augmentor_clip_output"])
            augmentor.to(self.device)
        elif self.augmentor_type == "transformer":
            augmentor = LpAugmentorTransformer(clip=self.config["augmentor_clip_output"])
            augmentor.to(self.device)
        else:
            raise ValueError(
                "Unrecognized augmentor type: {}".format(self.augmentor_type)
            )

        augmentor_optimizer = torch.optim.Adam(augmentor.parameters(), 3e-4)
        augmentor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            augmentor_optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )

        optimizer = torch.optim.Adam(
            list(model.parameters()),
            3e-4,
            weight_decay=eval(self.config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )

        if apex_support and self.config["fp16_precision"]:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config["epochs"]):
            print("====== Epoch {} =======".format(epoch_counter))
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._adv_step(model, augmentor, xis, xjs, n_iter)

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)

                if apex_support and self.config["fp16_precision"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # for p in augmentor.parameters():
                #     # print(p.name)
                #     p.grad *= -1.0
                optimizer.step()

                # Update augmentor
                augmentor_optimizer.zero_grad()
                loss = self._adv_step(model, augmentor, xis, xjs, n_iter)
                if self.augmentor_loss_type == "hinge":
                    loss = torch.clamp(loss, 0.0, 5.4)
                loss *= -1.0
                loss.backward()
                augmentor_optimizer.step()

                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, augmentor, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )
                print("validation loss: ", valid_loss)
                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
                augmentor_scheduler.step()

            self.writer.add_scalar(
                "cosine_lr_decay", scheduler.get_lr()[0], global_step=n_iter
            )

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./runs", self.config["fine_tune_from"], "checkpoints"
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, "model.pth"))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, augmentor, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._adv_step(model, augmentor, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss


class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config["batch_size"], **config["loss"]
        )

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "device" in self.config:
            device = self.config["device"]
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(
            model.parameters(), 3e-4, weight_decay=eval(self.config["weight_decay"])
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )

        if apex_support and self.config["fp16_precision"]:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config["epochs"]):
            print("epoch: ", epoch_counter)
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)

                if apex_support and self.config["fp16_precision"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )
                print("validation loss:", valid_loss)

                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar(
                "cosine_lr_decay", scheduler.get_lr()[0], global_step=n_iter
            )

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./runs", self.config["fine_tune_from"], "checkpoints"
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, "model.pth"))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
