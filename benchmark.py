#!/usr/bin/env python3
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


def read_array(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint16)


@dataclass
class RawMeta:
    name: str
    scene_id: str
    light: str
    ISO: int
    exp_time: float
    bayer_pattern: str
    shape: Tuple[int, int]
    wb_gain: Tuple[float, float, float]
    CCM: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    ROIs: List[Tuple[int, int, int, int]]


class BenchmarkLoader:

    def __init__(self, dataset_info_json: Path, base_path=None):
        with dataset_info_json.open() as f:
            self._dataset = [
                {
                    'input': x['input'],
                    'gt': x['gt'],
                    'meta': RawMeta(**x['meta'])
                }
                for x in json.load(f)
            ]
        if base_path is None:
            self.base_path = dataset_info_json.parent
        else:
            self.base_path = Path(base_path)

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, RawMeta]:
        if self._idx >= len(self):
            raise StopIteration

        input_bayer, gt_bayer, meta = self._load_idx(self._idx)

        self._idx += 1
        return input_bayer, gt_bayer, meta

    def _load_idx(self, idx: int):
        item = self._dataset[idx]

        img_files = item['input'], item['gt']
        meta = item['meta']
        bayers = []
        for img_file in img_files:
            if not Path(img_file).is_absolute():
                img_file = self.base_path / img_file
            bayer = read_array(img_file)
            bayer = bayer.reshape(*meta.shape)
            # Reno 10x outputs BGGR order
            assert meta.bayer_pattern == 'BGGR'
            bayer = bayer.astype(np.float32) / 65535
            bayers.append(bayer)

        input_bayer, gt_bayer = bayers
        return input_bayer, gt_bayer, meta

# vim: ts=4 sw=4 sts=4 expandtab
