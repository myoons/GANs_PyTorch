import os
import logging
import numpy as np
from tqdm import tqdm
from utils import denormalize

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TrainConfig:
    """ DCGAN Train Config """
    latent_size = None
    epochs = None
    batch_size = None
    beta_1 = None
    lr = None
    lr_decay = None

    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

        if not self.latent_size:
            self.latent_size = 100

        if not self.epochs:
            self.epochs = 200

        if not self.batch_size:
            self.batch_size = 128

        if not self.beta_1:
            self.beta_1 = 0.5

        if not self.lr:
            self.lr = 0.0002

        if not self.lr_decay:
            self.lr_decay = 0.99


class Trainer:

    def __init__(self, generator, discriminator, dataset, config):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.config = config

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.generator = torch.nn.DataParallel(self.generator.to(self.device))
            self.discriminator = torch.nn.DataParallel(self.discriminator.to(self.device))

        self.writer = SummaryWriter(config.ckpt_path)

    def save_checkpoint(self, name):
        raw_generator = self.generator.module if hasattr(self.generator, "module") else self.generator
        raw_discriminator = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator

        logger.info(f"saving {self.config.ckpt_path}/{name}")
        os.makedirs(f"{self.config.ckpt_path}/weights/{name}/", exist_ok=True)
        torch.save(raw_generator.state_dict(), f"{self.config.ckpt_path}/weights/{name}/G_{name}.ckpt")
        torch.save(raw_discriminator.state_dict(), f"{self.config.ckpt_path}/weights/{name}/D_{name}.ckpt")

    def log_tensorboard(self, path, item, step):
        self.writer.add_scalar(path, item, step)

    def configure_optimizers(self, generator, discriminator):
        optimizer_g = torch.optim.Adam(generator.parameters(),
                                       lr=self.config.lr,
                                       betas=(self.config.beta_1, 0.999))

        optimizer_d = torch.optim.Adam(discriminator.parameters(),
                                       lr=self.config.lr,
                                       betas=(self.config.beta_1, 0.999))

        return optimizer_g, optimizer_d

    def configure_lr_schedulers(self, optimizer_g, optimizer_d):
        scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g,
                                                        lr_lambda=lambda epoch: self.config.lr_decay ** epoch,
                                                        last_epoch=-1)

        scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d,
                                                        lr_lambda=lambda epoch: self.config.lr_decay ** epoch,
                                                        last_epoch=-1)

        return scheduler_g, scheduler_d

    def train(self):
        generator, discriminator, config = self.generator, self.discriminator, self.config

        criterion = torch.nn.BCELoss()
        optimizer_g, optimizer_d = self.configure_optimizers(generator, discriminator)
        scheduler_g, scheduler_d = self.configure_lr_schedulers(optimizer_g, optimizer_d)

        def run_epoch(epoch):

            generator.train()
            discriminator.train()
            loader = DataLoader(self.dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=6,
                                drop_last=True)

            losses_d, losses_g = [], []
            d_scores_real, d_scores_fake = [], []
            real_images, fake_images = None, None
            pbar = tqdm(enumerate(loader), total=len(loader))
            for step, real_images in pbar:
                batch_size = real_images.size(0)

                real_images = real_images.to(self.device)
                real_labels = torch.ones(batch_size).to(self.device)
                fake_labels = torch.zeros(batch_size).to(self.device)

                """ Train Discriminator """
                optimizer_d.zero_grad()

                real_outputs = discriminator(real_images).view(-1)
                d_scores_real.append(real_outputs.mean().item())
                real_loss_d = criterion(real_outputs, real_labels)

                z = torch.randn(batch_size, config.latent_size, 1, 1).to(self.device)
                fake_images = generator(z)
                fake_outputs = discriminator(fake_images).view(-1)
                fake_loss_d = criterion(fake_outputs, fake_labels)

                loss_d = real_loss_d + fake_loss_d
                losses_d.append(loss_d.item())
                loss_d.backward()
                optimizer_d.step()

                """ Train Generator """
                optimizer_g.zero_grad()

                z = torch.randn(batch_size, config.latent_size, 1, 1).to(self.device)
                fake_images = generator(z)
                fake_outputs = discriminator(fake_images).view(-1)
                d_scores_fake.append(fake_outputs.mean().item())

                loss_g = criterion(fake_outputs, real_labels)
                losses_g.append(loss_g.item())
                loss_g.backward()
                optimizer_g.step()

                pbar.set_description(f"Epoch [{epoch + 1}/{config.epochs}], Step [{step + 1}/{len(pbar)}], "
                                     f"Loss D : {loss_d.item():.4f}, Loss G : {loss_g.item():.4f}, "
                                     f"D(x) : {real_outputs.mean().item():.2f}, D(G(z)) : {fake_outputs.mean().item():.2f}")

                self.log_tensorboard("step/loss_D", loss_d.item(), epoch * len(loader) + step)
                self.log_tensorboard("step/loss_G", loss_g.item(), epoch * len(loader) + step)
                self.log_tensorboard("step/D_real", real_outputs.mean().item(), epoch * len(loader) + step)
                self.log_tensorboard("step/D_fake", fake_outputs.mean().item(), epoch * len(loader) + step)

            self.log_tensorboard("epoch/loss_D", np.mean(losses_d), epoch)
            self.log_tensorboard("epoch/loss_G", np.mean(losses_g), epoch)
            self.log_tensorboard("epoch/D_real", np.mean(d_scores_real), epoch)
            self.log_tensorboard("epoch/D_fake", np.mean(d_scores_fake), epoch)
            self.log_tensorboard("epoch/lr_D", optimizer_d.param_groups[0]['lr'], epoch)
            self.log_tensorboard("epoch/lr_G", optimizer_g.param_groups[0]['lr'], epoch)

            scheduler_d.step()
            scheduler_g.step()

            if epoch == 0:
                save_image(denormalize(real_images)[:64], os.path.join(f"{config.ckpt_path}", "images", "real_images.png"))

            save_image(denormalize(fake_images)[:64], os.path.join(f"{config.ckpt_path}", "images", f"fake_images_{epoch + 1}.png"))

            if (epoch+1) % 20 == 0:
                self.save_checkpoint(epoch + 1)

        for epc in range(config.epochs):
            run_epoch(epc)
