import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelConfig:
    """ DCGAN Model Config """
    latent_size = None
    negative_slope = None

    generator_conv1 = None
    generator_conv2 = None
    generator_conv3 = None
    generator_conv4 = None
    generator_conv5 = None

    discriminator_conv1 = None
    discriminator_conv2 = None
    discriminator_conv3 = None
    discriminator_conv4 = None
    discriminator_conv5 = None

    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

        if not self.latent_size:
            logger.error("latent_size is not implemented")
            raise NotImplementedError

        if not self.negative_slope:
            logger.error("negative_slop is not implemented")
            raise NotImplementedError

        if not self.generator_conv1:
            logger.error("generator_conv1 is not implemented")
            raise NotImplementedError

        if not self.generator_conv2:
            logger.error("generator_conv2 is not implemented")
            raise NotImplementedError

        if not self.generator_conv3:
            logger.error("generator_conv3 is not implemented")
            raise NotImplementedError

        if not self.generator_conv4:
            logger.error("generator_conv4 is not implemented")
            raise NotImplementedError

        if not self.generator_conv5:
            logger.error("generator_conv5 is not implemented")
            raise NotImplementedError

        if not self.discriminator_conv1:
            logger.error("discriminator_conv1 is not implemented")
            raise NotImplementedError

        if not self.discriminator_conv2:
            logger.error("discriminator_conv2 is not implemented")
            raise NotImplementedError

        if not self.discriminator_conv3:
            logger.error("discriminator_conv3 is not implemented")
            raise NotImplementedError

        if not self.discriminator_conv4:
            logger.error("discriminator_conv4 is not implemented")
            raise NotImplementedError

        if not self.discriminator_conv5:
            logger.error("discriminator_conv5 is not implemented")
            raise NotImplementedError


class Generator(nn.Module):
    """
    Generator with Transposed Convolution Layers
    input : Gaussian Random Noise z
    output : Generated Image
    """

    def __init__(self, config):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config.generator_conv1[0],
                               out_channels=config.generator_conv1[1],
                               kernel_size=config.generator_conv1[2],
                               stride=config.generator_conv1[3],
                               padding=config.generator_conv1[4]),
            nn.BatchNorm2d(config.generator_conv1[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config.generator_conv2[0],
                               out_channels=config.generator_conv2[1],
                               kernel_size=config.generator_conv2[2],
                               stride=config.generator_conv2[3],
                               padding=config.generator_conv2[4]),
            nn.BatchNorm2d(config.generator_conv2[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config.generator_conv3[0],
                               out_channels=config.generator_conv3[1],
                               kernel_size=config.generator_conv3[2],
                               stride=config.generator_conv3[3],
                               padding=config.generator_conv3[4]),
            nn.BatchNorm2d(config.generator_conv3[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config.generator_conv4[0],
                               out_channels=config.generator_conv4[1],
                               kernel_size=config.generator_conv4[2],
                               stride=config.generator_conv4[3],
                               padding=config.generator_conv4[4]),
            nn.BatchNorm2d(config.generator_conv4[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=config.generator_conv5[0],
                               out_channels=config.generator_conv5[1],
                               kernel_size=config.generator_conv5[2],
                               stride=config.generator_conv5[3],
                               padding=config.generator_conv5[4]),
            nn.Tanh()
        )

        self.apply(self.init_weights)

        logger.info(f"number of total parameters for G: {sum(p.numel() for p in self.parameters())}")
        logger.info(f"number of trainable parameters for G: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight, 0.0, 0.02)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            module.bias.data.zero_()

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    """
    Discriminator with Convolution Layers
    input : Image
    output : 0~1 float (0: Fake Image, 1: Real Image)
    """

    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=config.discriminator_conv1[0],
                      out_channels=config.discriminator_conv1[1],
                      kernel_size=config.discriminator_conv1[2],
                      stride=config.discriminator_conv1[3],
                      padding=config.discriminator_conv1[4]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv2[0],
                      out_channels=config.discriminator_conv2[1],
                      kernel_size=config.discriminator_conv2[2],
                      stride=config.discriminator_conv2[3],
                      padding=config.discriminator_conv2[4]),
            nn.BatchNorm2d(config.discriminator_conv2[1]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv3[0],
                      out_channels=config.discriminator_conv3[1],
                      kernel_size=config.discriminator_conv3[2],
                      stride=config.discriminator_conv3[3],
                      padding=config.discriminator_conv3[4]),
            nn.BatchNorm2d(config.discriminator_conv3[1]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv4[0],
                      out_channels=config.discriminator_conv4[1],
                      kernel_size=config.discriminator_conv4[2],
                      stride=config.discriminator_conv4[3],
                      padding=config.discriminator_conv4[4]),
            nn.BatchNorm2d(config.discriminator_conv4[1]),
            nn.LeakyReLU(config.negative_slope),
            nn.Conv2d(in_channels=config.discriminator_conv5[0],
                      out_channels=config.discriminator_conv5[1],
                      kernel_size=config.discriminator_conv5[2],
                      stride=config.discriminator_conv5[3],
                      padding=config.discriminator_conv5[4]),
            nn.Sigmoid()
        )

        logger.info(f"number of total parameters for D: {sum(p.numel() for p in self.parameters())}")
        logger.info(f"number of trainable parameters for D: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            module.bias.data.zero_()

    def forward(self, x):
        return self.discriminator(x)
