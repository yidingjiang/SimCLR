import torch
from models.resnet_simclr import ResNetSimCLR
from models.icm_augmentor import IcmAugmentor
from models.icm_augmentor import Discriminator
from models.icm_augmentor import IcmAugmentorv2
from models.icm_augmentor import Discriminatorv2
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


class IcmSimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config["batch_size"], **config["loss"]
        )
        self.dis_criterion = torch.nn.CrossEntropyLoss()
        self.normal_dist = tdist.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        self.disc_weight = 0.1

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "device" in self.config:
            device = self.config["device"]
        print("Running on:", device)
        return device

    def _step(self, model, augmentor, xis, xjs, n_iter):
        # shape = augmentor.noise_shapes(eval(self.config["dataset"]["input_shape"])[0])
        xis_o, xjs_o = xis, xjs

        xis, xis_mech_label = augmentor(xis)
        xjs, xjs_mech_label = augmentor(xjs)

        ris, zis = model(xis)  # [N,C]
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def _disc_step(self, augmentor, discriminator, xis, xjs, n_iter):
        xis_o, xjs_o = xis, xjs
        xis, xis_mech_label = augmentor(xis)
        xis_prediction = discriminator((xis, xis_o))
        xjs, xjs_mech_label = augmentor(xjs)
        xjs_prediction = discriminator((xjs, xjs_o))
        xis_mech_label, xjs_mech_label = np.int32(xis_mech_label[0]), np.int32(
            xjs_mech_label[0]
        )
        disc_loss_i = self.dis_criterion(
            xis_prediction, torch.Tensor(xis_mech_label).long().to(self.device)
        )
        disc_loss_j = self.dis_criterion(
            xjs_prediction, torch.Tensor(xjs_mech_label).long().to(self.device)
        )
        total_loss = (disc_loss_i + disc_loss_j) / 2.0

        if n_iter % 100 == 0:
            print("step{}    D loss: {:6f}".format(n_iter, total_loss))

        return total_loss

    def _build_model(self):
        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        augmentor = IcmAugmentor(num_mech=self.config["num_mechanisms"]).to(self.device)
        augmentor = self._load_pre_trained_weights(augmentor, "augmentor.pth")

        discriminator = Discriminator(num_mech=self.config["num_mechanisms"]).to(self.device)
        discriminator = self._load_pre_trained_weights(discriminator, "discriminator.pth")
        return model, augmentor, discriminator

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model, augmentor, discriminator = self._build_model()

        optimizer = torch.optim.Adam(
            list(model.parameters()),
            3e-4,
            weight_decay=eval(self.config["weight_decay"]),
        )

        aug_optimizer = torch.optim.Adam(
            list(augmentor.parameters()),
            3e-4,
            weight_decay=eval(self.config["weight_decay"]),
        )

        disc_optimizer = torch.optim.Adam(
            list(discriminator.parameters()),
            3e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )
        aug_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            aug_optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )
        disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            disc_optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
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
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                if n_iter % self.config["simclr_train_interval"] == 0:
                    optimizer.zero_grad()

                    loss = self._step(model, augmentor, xis, xjs, n_iter)

                    if n_iter % self.config["log_every_n_steps"] == 0:
                        self.writer.add_scalar("train_loss", loss, global_step=n_iter)

                    if apex_support and self.config["fp16_precision"]:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                # Update augmentor
                aug_optimizer.zero_grad()
                loss = -self._step(model, augmentor, xis, xjs, n_iter)
                loss += self.disc_weight * self._disc_step(
                    augmentor, discriminator, xis, xjs, n_iter
                )
                loss.backward()
                aug_optimizer.step()

                # Update Discriminator
                disc_optimizer.zero_grad()
                loss = self._disc_step(augmentor, discriminator, xis, xjs, n_iter)
                loss.backward()
                disc_optimizer.step()

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
                    torch.save(
                        augmentor.state_dict(),
                        os.path.join(model_checkpoints_folder, "augmentor.pth"),
                    )
                    torch.save(
                        discriminator.state_dict(),
                        os.path.join(model_checkpoints_folder, "discriminator.pth"),
                    )
                print("validation loss: ", valid_loss)
                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
                aug_scheduler.step()
                disc_scheduler.step()

            self.writer.add_scalar(
                "cosine_lr_decay", scheduler.get_lr()[0], global_step=n_iter
            )

    def _load_pre_trained_weights(self, model, model_path="model.pth"):
        try:
            checkpoints_folder = os.path.join(
                "./runs", self.config["fine_tune_from"], "checkpoints"
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, model_path))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model from with success.".format(model_path))
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

                loss = self._step(model, augmentor, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss


class IcmSimCLRv2(IcmSimCLR):

    def __init__(self, dataset, config):
         super(IcmSimCLRv2, self).__init__(dataset, config)

    def _build_model(self):
        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        augmentor = IcmAugmentorv2(num_mech=self.config["num_mechanisms"]).to(self.device)
        augmentor = self._load_pre_trained_weights(augmentor, "augmentor.pth")

        discriminator = Discriminatorv2(num_mech=self.config["num_mechanisms"]).to(self.device)
        discriminator = self._load_pre_trained_weights(discriminator, "discriminator.pth")
        return model, augmentor, discriminator

    def _disc_step(self, augmentor, discriminator, xis, xjs, n_iter):
        xis_o, xjs_o = xis, xjs

        xis, xis_mech_label = augmentor(xis)
        xis_pred_id, xis_pred_val = discriminator((xis, xis_o))
        xjs, xjs_mech_label = augmentor(xjs)
        xjs_pred_id, xjs_pred_val = discriminator((xjs, xjs_o))

        xis_true_id = np.int32(xis_mech_label['id'][0])
        xjs_true_id = np.int32(xjs_mech_label['id'][0])
        xis_true_val = np.float32(xis_mech_label['value'][0])
        xjs_true_val = np.float32(xjs_mech_label['value'][0])

        id_loss_i = self.dis_criterion(
            xis_pred_id, torch.Tensor(xis_true_id).long().to(self.device)
        )
        id_loss_j = self.dis_criterion(
            xjs_pred_id, torch.Tensor(xjs_true_id).long().to(self.device)
        )
        total_id_loss = (id_loss_i + id_loss_j) / 2.

        val_loss_i = torch.mean(
            (xis_pred_val - torch.Tensor(xis_true_val).float().to(self.device))**2
        )
        val_loss_j = torch.mean(
            (xjs_pred_val - torch.Tensor(xjs_true_val).float().to(self.device))**2
        )
        total_val_loss = (val_loss_i + val_loss_j) / 2.

        if n_iter % 100 == 0:
            print("step{}    id loss: {:6f}    val_loss: {:6f}".format(n_iter, total_id_loss, total_val_loss))

        return total_id_loss + total_val_loss
