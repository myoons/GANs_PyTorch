import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelConfig:
    """ Vanilla GAN Model Config """
    latent_size = None
    discriminator_first_hidden_size = None
    discriminator_second_hidden_size = None
    discriminator_dropout = None
    generator_first_hidden_size = None
    generator_second_hidden_size = None
    generator_dropout = None
    negative_slope = None
    image_size = None

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


class Generator(nn.Module):
    """
    Generator with Linear Layers
    input : Gaussian Random Noise z
    output : Generated Image
    """

    def __init__(self, config):
        super(Generator, self).__init__()

        assert int(config.image_size ** 0.5) ** 2 == config.image_size, "image size should be square number"
        self.image_len = int(config.image_size ** 0.5)

        self.generator = nn.Sequential(
            nn.Linear(config.latent_size, config.generator_first_hidden_size),
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
        logger.info(f"number of trainable parameters for G: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.generator(x).view(-1, 1, self.image_len, self.image_len)


class Discriminator(nn.Module):
    """
    Discriminator with Linear Layers
    input : Image
    output : 0~1 float (0: Fake Image, 1: Real Image)
    """

    def __init__(self, config):
        super(Discriminator, self).__init__()

        assert int(config.image_size ** 0.5) ** 2 == config.image_size, "Image Size should be Square Number"
        self.image_size = config.image_size

        self.discriminator = nn.Sequential(
            nn.Linear(config.image_size, config.discriminator_first_hidden_size),
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
        logger.info(f"number of trainable parameters for D: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.discriminator(x.view(-1, self.image_size))
