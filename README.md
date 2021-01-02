# Practical Mobile Raw Image Denoising (PMRID)

Code and dataset for ECCV20 paper [Practical Deep Raw Image Denoising on Mobile Devices](https://arxiv.org/abs/2010.06935).

## Dataset

### Downloads
- [OneDrive](https://megvii-my.sharepoint.cn/:f:/g/personal/wangyuzhi_megvii_com/Et4v2Z7CkRxHnbcFUq6RXZMBfXUrlm_Se5OVDvcdujVsMA?e=vcfJWs)
- [Kaggle](https://www.kaggle.com/dataset/1bdc5cd707cfbb3ee842eb3cbfe93495dbba88017d29f295f8edbcb8f8790556)

### Usage

The dataset includes two 7zip files:
- `reno10x_noise.7z` contains DNG raw images shot by an _OPPO Reno 10x_ phone for noise parameter estimation (refer Sec 3.1 and 5.1 in the paper)
- `PMRID.7z` is the benchmark dataset described in Sec 5.2 in the paper

The structure of `PMRID.7z` is
```
- benchmark.json  # meta info
- Scene1/
  \- Bright/
     \- exposure-case1/ 
         \- input.raw   # RAW data for noisy image in uint16
          - gt.raw      # RAW data for clean image in uint16
      + case2/
  + Dark/
+ Secne2/
```

All metadata for images are listed in `benchmark.json`:
```python
{
   "input": "path/to/noisy_input.raw",
   "gt": "path/to/clean_gt.raw",
   "meta": {
       "name": "case_name",
       "scene_id": "scene_name",
       "light": "light condition",
       "ISO": "ISO",
       "exp_time": "exposure time",
       "bayer_pattern": "BGGR",
       "shape": [3000, 4000],
       "wb_gain": [r_gain, g_gain, b_gain],
       "CCM": [   # 3x3 color correction matrix
           [c11, c12, c13], 
           [c21, c22, c23], 
           [c31, c32, c33]
       ],
       "ROIs": [  # patch ROIs to calculate PSNR and SSIM, x0 is topleft
           [topleft_w, topleft_h, bottomright_w, bottomright_h]
       ]
   }
}
```

## Pre-trained Models and Benchmark Script

Both [PyTorch](https://pytorch.org/) and [MegEngine](https://megengine.org.cn/) pre-trained models are provided in the `models` directory. 
The benchmark script is written for models trained with MegEngine. `Python >= 3.6` is required to run the benchmark script.

```
pip install -r requirements.txt
python3 run_benchmark.py --benchmark /path/to/PMRID/benchmark.json models/mge_pretrained.ckp
```


## Citation
```
@inproceedings{wang2020,
	title={Practical Deep Raw Image Denoising on Mobile Devices},
	author={Wang, Yuzhi and Huang, Haibin and Xu, Qin and Liu, Jiaming and Liu, Yiqun and Wang, Jue},
	booktitle={European Conference on Computer Vision (ECCV)},
	year={2020},
	pages={1--16}
}
```
