#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import megengine as mge
import numpy as np
import skimage.metrics
from tqdm import tqdm

from models.net_mge import Network
from utils import RawUtils
from dataset.benchmark import BenchmarkLoader, RawMeta


class KSigma:

    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float, float], anchor: float, V: float = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V


class Denoiser:

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=256.0):
        net = Network()
        with model_path.open('rb') as f:
            states = pickle.load(f)
        net.load_state_dict(states)
        net.eval()

        self.net = net
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01: np.ndarray):
        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)

        H, W = rggb.shape[:2]
        ph, pw = (32-(H % 32))//2, (32-(W % 32))//2
        rggb = np.pad(rggb, [(ph, ph), (pw, pw), (0, 0)], 'constant')
        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale

        inp = np.ascontiguousarray(inp_rggb)
        pred = self.net(inp)[0] / self.inp_scale

        # import ipdb; ipdb.set_trace()
        pred = pred.numpy().transpose(1, 2, 0)
        pred = self.ksigma(pred, iso, inverse=True)

        ph, pw = self.ph, self.pw
        pred = pred[ph:-ph, pw:-pw]
        return RawUtils.rggb2bayer(pred)


def run_benchmark(model_path, bm_loader: BenchmarkLoader):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = Denoiser(model_path, ksigma)

    PSNRs, SSIMs = [], []

    bar = tqdm(bm_loader)
    for input_bayer, gt_bayer, meta in bar:
        bar.set_description(meta.name)
        assert meta.bayer_pattern == 'BGGR'
        input_bayer, gt_bayer = RawUtils.bggr2rggb(input_bayer, gt_bayer)

        pred_bayer = denoiser.run(input_bayer, iso=meta.ISO)

        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
            input_bayer, pred_bayer, gt_bayer,
            wb_gain=meta.wb_gain, CCM=meta.CCM,
        )
        inp_rgb, pred_rgb, gt_rgb = RawUtils.bggr2rggb(inp_rgb, pred_rgb, gt_rgb)
        bar.set_description(meta.name+' ✓')

        psnrs = []
        ssims = []

        for x0, y0, x1, y1 in meta.ROIs:
            pred_patch = pred_rgb[y0:y1, x0:x1]
            gt_patch = gt_rgb[y0:y1, x0:x1]

            psnr = skimage.metrics.peak_signal_noise_ratio(gt_patch, pred_patch)
            ssim = skimage.metrics.structural_similarity(gt_patch, pred_patch, multichannel=True)
            psnrs.append(float(psnr))
            ssims.append(float(ssim))

        bar.set_description(meta.name+' ✓✓')

        PSNRs = PSNRs + psnrs   # list append
        SSIMs = SSIMs + ssims

    mean_psnr = np.mean(PSNRs)
    mean_ssim = np.mean(SSIMs)
    print("mean PSNR:", mean_psnr)
    print("mean SSIM:", mean_ssim)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--benchmark', type=Path)

    args = parser.parse_args()

    bm_loader = BenchmarkLoader(args.benchmark.resolve())
    run_benchmark(args.model, bm_loader)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
