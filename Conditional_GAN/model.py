import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelConfig:
    """ Conditional GAN Model Config """
    latent_size = None
    discriminator_first_hidden_size = None
    discriminator_second_hidden_size = None
    discriminator_dropout = None
    generator_first_hidden_size = None
    generator_second_hidden_size = None
    generator_dropout = None
    negative_slope = None
    image_size = None
    n_classes = None
    label_embed_size = None

    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

        if not self.latent_size:
            logger.error("latent_size is not implemented")
            raise NotImplementedError

        if not self.discriminator_first_hidden_size:
            logger.error("discriminator_first_hidden_size is not implemented")
            raise NotImplementedError

        if not self.discriminator_second_hidden_size:
            logger.error("discriminator_second_hidden_size is not implemented")
            raise NotImplementedError

        if not self.discriminator_dropout:
            logger.error("discriminator_dropout is not implemented")
            raise NotImplementedError

        if not self.generator_first_hidden_size:
            logger.error("generator_first_hidden_size is not implemented")
            raise NotImplementedError

        if not self.generator_second_hidden_size:
            logger.error("generator_second_hidden_size is not implemented")
            raise NotImplementedError

        if not self.generator_dropout:
            logger.error("generator_dropout is not implemented")
            raise NotImplementedError

        if not self.negative_slope:
            logger.error("negative_slope is not implemented")
            raise NotImplementedError

        if not self.image_size:
            logger.error("image_size is not implemented")
            raise NotImplementedError

        if not self.n_classes:
            logger.error("n_classes is not implemented")
            raise NotImplementedError

        if not self.label_embed_size:
            logger.error("label_embed_size is not implemented")
            raise NotImplementedError


class Generator(nn.Module):
    """
    Generator with Linear Layers and Condition
    input : Gaussian Random Noise z
    output : Generated Image
    """

    def __init__(self, config):
        super(Generator, self).__init__()

        self.image_len = int(config.image_size ** 0.5)

        self.label_embed = nn.Embedding(config.n_classes, config.label_embed_size)

        self.generator = nn.Sequential(
            nn.Linear(config.latent_size + config.label_embed_size, config.generator_first_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.generator_dropout),
            nn.Linear(config.generator_first_hidden_size, config.generator_second_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.generator_dropout),
            nn.Linear(config.generator_second_hidden_size, config.image_size),
            nn.Tanh()
        )

        self.apply(self.init_weights)

        logger.info(f"number of total parameters for G: {sum(p.numel() for p in self.parameters())}")
        logger.info(
            f"number of trainable parameters for G: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, x, y):
        x = torch.cat([self.label_embed(y), x], dim=-1)
        return self.generator(x).view(-1, 1, self.image_len, self.image_len)


class Discriminator(nn.Module):
    """
    Discriminator with Linear Layers and Condition
    input : Image
    output : 0~1 float (0: Fake Image, 1: Real Image)
    """

    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.image_size = config.image_size

        self.label_embed = nn.Embedding(config.n_classes, config.label_embed_size)

        self.discriminator = nn.Sequential(
            nn.Linear(config.image_size + config.label_embed_size, config.discriminator_first_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.discriminator_dropout),
            nn.Linear(config.discriminator_first_hidden_size, config.discriminator_second_hidden_size),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.discriminator_dropout),
            nn.Linear(config.discriminator_second_hidden_size, 1),
            nn.Sigmoid()
        )

        self.apply(self.init_weights)

        logger.info(f"number of parameters for D: {sum(p.numel() for p in self.parameters())}")
        logger.info(
            f"number of trainable parameters for D: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, x, y):
        x = torch.cat([self.label_embed(y), x.view(-1, self.image_size), ], dim=-1)
        return self.discriminator(x)
