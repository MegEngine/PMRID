import json
import math
from enum import Enum
from typing import Optional, List, Tuple

import numpy as np
import megengine as mge
import megengine.random
import megengine.functional as F

from pydantic import BaseModel
from megfile import SmartPath, smart_load_from
from megengine.data.dataset import Dataset


class BayerPattern(Enum, str):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


class RawImageItem(BaseModel):
    path: str
    width: int
    height: int
    black_level: int
    white_level: int = 65535
    bayer_pattern: BayerPattern
    g_mean_01: float


class NoiseProfile(BaseModel):
    K: Tuple[float, float]
    B: Tuple[float, float, float]
    value_scale: float = 959.0


class DataAugOptions(BaseModel):
    noise_profile: NoiseProfile
    camera_value_scale: float = 959.0
    iso_range: Tuple[float, float]
    anchor_iso: float = 1600.0
    output_shape: Tuple[int, int] = (512, 512)   # 512x512x4
    target_brighness_range: Tuple[float, float] = (0.02, 0.5)


class CleanRawImages(Dataset):

    def __init__(self, *, index_file: Optional[str], data_dir: Optional[SmartPath], opts: DataAugOptions):
        """
        Args:
            - data_dir: a directory that contains "index.json" and raw images
            - index_file: the absolute path to the index file
        """
        super().__init__()

        assert not (index_file is None and data_dir is None)

        if data_dir is None:
            index_file = SmartPath(index_file)
        else:
            assert index_file is None
            index_file = data_dir / "index.json"

        self.opts = DataAugOptions
        self.filelist: List[RawImageItem] = []
        with index_file.open() as f:
            items = [RawImageItem.parse_obj(x) for x in json.load(f)]
            for item in items:
                if data_dir is not None:
                    item.path = str(data_dir / item.path)
                self.filelist.append(item)

    def __len__(self):
        return len(self.filelist)

    def random_flip_and_crop(self, img: np.ndarray, src_bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Random flip and crop a bayter-patterned image, and normalize the bayer pattern to RGGB.
        """

        flip_ud = np.random.rand() > 0.5
        flip_lr = np.random.rand() > 0.5

        if src_bayer_pattern == BayerPattern.RGGB:
            crop_x_offset, crop_y_offset = 0, 0
        elif src_bayer_pattern == BayerPattern.GBRG:
            crop_x_offset, crop_y_offset = 0, 1
        elif src_bayer_pattern == BayerPattern.GRBG:
            crop_x_offset, crop_y_offset = 1, 0
        elif src_bayer_pattern == BayerPattern.BGGR:
            crop_x_offset, crop_y_offset = 1, 1

        if flip_lr:
            crop_x_offset = (crop_x_offset + 1) % 2
        if flip_ud:
            crop_y_offset = (crop_y_offset + 1) % 2

        H0, W0 = img.shape
        tH, tW = self.opts.output_shape

        x0, y0 = np.random.randint(0, W0 - tW), np.random.randint(0, H0 - tH)
        x0, y0 = x0 // 2 * 2 + crop_x_offset, y0 // 2 * 2 + crop_y_offset

        img_crop = img[y0:y0+tH, x0:x0+tW]
        if flip_lr:
            img_crop = np.flip(img_crop, axis=1)
        if flip_ud:
            img_crop = np.flip(img_crop, axis=0)

        return img_crop

    def __getitem__(self, index: int):
        item = self.filelist[index]
        buf = smart_load_from(item.path)
        rawimg = np.fromfile(buf, dtype=np.uint16).reshape((item.height, item.width))
        # random crop to output size
        rawimg = self.random_flip_and_crop(rawimg, item.bayer_pattern).astype(np.float32)

        raw01 = (rawimg - item.black_level) / (item.white_level - item.black_level)
        H, W = raw01.shape
        # pixel shuffle to RGGB image
        rggb01 = raw01.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)
        return rggb01, np.array(item.g_mean_01)


class NoiseProfileFunc:

    def __init__(self, noise_profile: NoiseProfile):
        self.polyK = np.poly1d(noise_profile.K)
        self.polyB = np.poly1d(noise_profile.B)
        self.value_scale = noise_profile.value_scale

    def __call__(self, iso, value_scale=959.0):
        r = value_scale / self.value_scale
        k = self.polyK(iso) * r
        b = self.polyB(iso) * r * r

        return k, b


class DataAug:

    def __init__(self, opts: DataAugOptions):
        self.opts = opts
        self.noise_func = NoiseProfileFunc(opts.noise_profile)

    def transform(self, batch_img01: np.ndarray, batch_g_mean: float) -> Tuple[mge.Tensor, mge.Tensor, mge.Tensor]:
        """
        Args:
            - img: [-black/camera_value_scale, 1.0]

        Returns:
            - noisy_img
            - iso
        """

        batch_imgs = mge.tensor(batch_img01) * self.opts.camera_value_scale
        batch_gt = self.brightness_aug(batch_imgs, batch_g_mean)
        batch_imgs, batch_iso = self.add_noise(batch_gt)
        cvt_k, cvt_b = self.k_sigma(batch_iso)

        batch_imgs = batch_imgs * cvt_k + cvt_b
        batch_gt = batch_gt * cvt_k + cvt_b
        return (batch_imgs, batch_gt, cvt_k)

    def k_sigma(self, iso: float) -> Tuple[float, float]:
        k, sigma = self.noise_func(iso, value_scale=self.opts.camera_value_scale)
        k_a, sigma_a = self.noise_func(self.opts.anchor_iso, value_scale=self.opts.camera_value_scale)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        return cvt_k, cvt_b

    def brightness_aug(self, img_batch: mge.Tensor, orig_gmean: List[float]) -> mge.Tensor:
        low, high = self.opts.target_brighness_range
        N = len(orig_gmean)
        btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N, )))
        s = np.clip(btarget / orig_gmean, 0.01, 1.0)
        return img_batch * s.reshape(-1, 1, 1, 1)

    def add_noise(self, img: mge.Tensor) -> Tuple[mge.Tensor, float]:
        """
        Args:
            - img: [-black, camera_value_scale]

        Returns:
            - noisy_img
            - iso
        """

        N = img.shape[0]
        isos = np.random.uniform(*self.opts.iso_range, size=(N, ))
        k, b = self.noise_func(isos, value_scale=self.opts.camera_value_scale)
        k = k.reshape(-1, 1, 1, 1)
        b = b.reshape(-1, 1, 1, 1)

        shot_noisy = megengine.random.poisson((img / k).clip(0, 1)) * k
        read_noisy = megengine.random.normal(size=img.shape) * math.sqrt(b)
        noisy = shot_noisy + read_noisy
        noisy = F.round(noisy)

        return noisy