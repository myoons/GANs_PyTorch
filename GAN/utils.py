import os

import torch
import random
import datetime
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(images):
    out = (images + 1) / 2
    return out.clamp(0, 1)


def make_ckpt_directory(train_config):
    path = f"checkpoints/{datetime.datetime.now()}"
    os.makedirs(path)
    os.makedirs(f'{path}/images/')
    os.makedirs(f'{path}/weights/')
    train_config['ckpt_path'] = path
