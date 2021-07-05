import yaml
import logging
from utils import set_seed, make_ckpt_directory
from trainer import TrainConfig, Trainer
from model import ModelConfig, Generator, Discriminator
from dataset import SimpleDataset

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == '__main__':
    set_seed(42)

    images = np.load('data/mnist_images.npy')
    dataset = SimpleDataset(images)
    myaml = yaml.load(open('config/model.yaml', 'r'), Loader=yaml.FullLoader)
    mconf = ModelConfig(myaml)

    generator = Generator(mconf)
    discriminator = Discriminator(mconf)

    tyaml = yaml.load(open(f'config/train.yaml', 'r'), Loader=yaml.FullLoader)
    make_ckpt_directory(tyaml)
    tconf = TrainConfig(tyaml)

    trainer = Trainer(generator, discriminator, dataset, tconf)
    trainer.train()
