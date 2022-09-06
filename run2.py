import argparse
import os

from model.srgan import Discriminator, GeneratorResNet, FeatureExtractor

os.environ()
import sys
import warnings

import numpy as np
import config.default as C

import logging

import torch
from torch.optim import Adam

from tqdm import tqdm

from src.utils.fix_seed import seed_everything

# global args(configuration)
args = None


def parse_args():
    parser = argparse.ArgumentParser(description="SRGAN written by Han")

    # data root
    parser.add_argument("--root", type=str)

    # environment
    parser.add_argument("--gpus", default=C.DEVICE, type=str)
    parser.add_argument("--cpus", default=8, type=int)

    # hyper parameters
    parser.add_argument("--seed", default=C.SEED, type=int)
    parser.add_argument("--epochs", default=C.EPOCH, type=int)
    parser.add_argument("--batch-size", default=C.BATCH_SIZE, type=int)
    parser.add_argument("--base-lr", default=C.LR, type=float)
    parser.add_argument("--milestones", default=C.MILESTONES, type=str)
    parser.add_argument("--warmup", default=C.WARM_UP)
    
    # model parameter
    parser.add_argument("--b1", default=0.5, type=float)
    parser.add_argument("--b2", default=0.999, type=float)
    parser.add_argument("--decay_epoch", default=100, type=int)

    # data parameter
    parser.add_argument("--hr_height", default=256, type=int)
    parser.add_argument("--hr_width", default=256, type=int)
    parser.add_argument("--channels", default=3, type=int)

    # log parameter
    parser.add_argument("--log-dir", default=f"./logs/{C.DAY}/{C.TIME}", type=str)
    parser.add_argument("--resume", default=5, type=int)

    global args
    args = parser.parse_args()
    
    # add other arguments
    args.device_id = [int(d) for d in args.gpus.split(",")]
    args.milestone = [int(m) for m in args.milestones.split(",")]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # make directory
    os.makedirs(os.path.join(args.log_dir, "image_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "model_logs"), exist_ok=True)
    
    
def print_args():
    for arg in vars(args):
        s = f"{arg}: {getattr(args, arg)}"
        logging.info(s)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info(f"\nSave checkpoint to {filename}")


def adjust_learning_rate(optimizer, epoch, milestones=None):

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    lr = args.base_lr * (0.2**n)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

hr_shape = (args.hr_height, args.hr_width)

generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(args.channel, *hr_shape))
feature_extractor = FeatureExtractor()

optimizer_G = Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = Adam(discriminator, lr=args.lr, betas=(args.b1, args.b2))

train_loader, valid_loader = 


def train(train_loader):
    for idx, features in enumerate(tqdm(train_loader)):
        optimizer_G.


def main():
    parse_args()
    seed_everything(args.seed)
    
    # logging step
    logging.basicConfig(
        format="[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.log_dir, f"{C.DAY}-{C.TIME}.log"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )
    
    print_args(args)  # print args