#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import megengine as mge
import megengine.optimizer
import megengine.functional as F
from megengine.data import DataLoader, RandomSampler
from megengine.autodiff import GradManager

from tqdm import tqdm
from loguru import logger

from models.net_mge import Network, get_loss_l1
from dataset.training import CleanRawImages, DataAug, DataAugOptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-aug-config', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--ckp-dir', default=Path('./checkpoints'), type=Path)
    parser.add_argument('--learning-rate', dest='lr', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=4000, type=int)

    args = parser.parse_args()

    # Configure loggger
    logger.configure(handlers=[dict(
        sink=lambda msg: tqdm.write(msg, end=''),
        format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] [<level>{level}</level>] {message}",
        colorize=True
    )])

    # Create model
    net = Network()
    # Create optimizer
    optimizer = megengine.optimizer.Adam(net.parameters(), lr=args.lr)
    # Create GradManager
    gm = GradManager().attach(net.parameters())

    aug_opts = DataAugOptions.parse_file(args.data_aug_config)
    train_aug = DataAug(aug_opts)
    train_ds = CleanRawImages(data_dir=args.data_dir, opts=aug_opts)
    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds, batch_size=args.batch_size, drop_last=True))

    # learning rate scheduler
    def adjust_learning_rate(opt, epoch, step):
        M = len(train_ds) // args.batch_size
        T = M * 100
        Th = T // 2

        # # warm up
        # if base_lr > 2e-3 and step < T:
        #     return 1e-4

        if epoch < 3000:
            f = 1 - step / (M*3000)
        elif epoch < 3000:
            f = 0.1
        elif epoch < 5000:
            f = 0.2
        else:
            f = 0.1

        t = step % T
        if t < Th:
            f2 = t / Th
        else:
            f2 = 2 - (t/Th)

        lr = f * f2 * args.lr

        for pgroup in opt.param_groups:
            pgroup["lr"] = lr

        return lr

    # train step
    def train_step(img, gt, norm_k):
        with gm:
            pred = net(img)
            loss = get_loss_l1(pred, gt, norm_k)
            gm.backward(loss)
        optimizer.step().clear_grad()
        return loss

    # train loop
    global_step = 0
    for epoch in range(args.num_epoch):
        for bidx, (imgs, g_means) in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            imgs, gt, norm_k = train_aug.transform(imgs, g_means)
            lr = adjust_learning_rate(optimizer, epoch, global_step)
            loss = train_step(imgs, gt, norm_k)

            if global_step % 100 == 0:
                logger.info(f"clock: {epoch},{bidx}, loss: {loss.item()}, lr: {lr}")

            global_step += 1

        # save checkpoint
        if epoch % 100 == 0:
            mge.save(net.state_dict(), args.ckp_dir / f"epoch_{epoch}.pkl")


if __name__ == "__main__":
    main()