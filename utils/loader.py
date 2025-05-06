# nano_dataloader.py
"""
NanoDataLoader
--------------

A PyTorch‑Lightning DataModule for nanoscale grayscale images (1700 × 1600).
Outputs for every mini‑batch:
  • LR tensor  [B, 3, H//r, W//r]
  • HR tensor  [B, 8, H, W]          (channels: 3×gray, albedo, gaussian, sobel‑depth, normal‑variation, fourier)
  • Fourier mask tensor [B, 1, H, W]
"""

# ──────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────
import os
import random
from typing import List, Optional

import cv2 as cv
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

# ──────────────────────────────────────────────────────────────────
# -----  Mask‑generation helpers (single‑sample versions)  ---------
# ──────────────────────────────────────────────────────────────────
def _pil_to_np_gray(pil: Image.Image) -> np.ndarray:
    """PIL → float32 numpy array in [0,1], shape (H, W)."""
    arr = np.asarray(pil.convert("L"), dtype=np.float32)
    return cv.normalize(arr, None, 0.0, 1.0, cv.NORM_MINMAX)

def _albedo(gray: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    smooth = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return (np.abs(gray - smooth) > thresh).astype(np.float32)

def _gaussian(gray: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    smooth = cv.GaussianBlur(gray, (9, 9), 1.5)
    return (np.abs(gray - smooth) > thresh).astype(np.float32)

def _sobel_magnitude(gray: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return (mag > thresh).astype(np.float32)

def _normal_variation(gray: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    direction = np.arctan2(gy, gx)
    dx = cv.Sobel(direction, cv.CV_32F, 1, 0, ksize=3)
    dy = cv.Sobel(direction, cv.CV_32F, 0, 1, ksize=3)
    var = np.sqrt(dx**2 + dy**2)
    return (var > thresh).astype(np.float32)

def _fourier(gray: np.ndarray, perc: int = 95) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    thr = np.percentile(mag, perc)
    high_freq_mask = (mag >= thr).astype(np.float32)
    f_filtered = f * high_freq_mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    img_back = cv.normalize(img_back, None, 0, 1, cv.NORM_MINMAX)
    return (img_back > 0.1).astype(np.float32)

# ──────────────────────────────────────────────────────────────────
# -------------------      Dataset Class       --------------------
# ──────────────────────────────────────────────────────────────────
class _NanoDataset(Dataset):
    """
    单张灰度 TIFF/PNG/JPG/EXR → LR / HR‑8c / Fourier mask
    """
    def __init__(self,
                 image_paths: List[str],
                 r: int,
                 high_res: int):
        super().__init__()
        self.image_paths = image_paths
        self.r = r
        self.hr_size = high_res          # e.g. 256
        self.crop_origin_max = 1600 - high_res  # 1600×1600 after trimming bottom

    # ──────────────────────────────────────────
    def __len__(self):
        return len(self.image_paths)

    # ──────────────────────────────────────────
    def __getitem__(self, idx):
        # ---- 读图 & 裁剪底部 ----
        path = self.image_paths[idx]
        pil = Image.open(path)                     # 灰度 / 彩色皆可
        pil = pil.convert("L")                    # 强制灰度
        pil = pil.crop((0, 0, 1600, 1600))        # 去掉底部 100 行 (1700→1600)

        # ---- 随机 crop high_res×high_res ----
        x0 = random.randint(0, self.crop_origin_max)
        y0 = random.randint(0, self.crop_origin_max)
        pil_patch = pil.crop((x0, y0, x0 + self.hr_size, y0 + self.hr_size))

        gray_np = _pil_to_np_gray(pil_patch)       # (H, W) ∈ [0,1]

        # ---- 生成 3‑通道 Gray & 各种 mask ----
        gray_3 = np.stack([gray_np]*3, axis=0)     # [3,H,W]

        albedo     = _albedo(gray_np)[None, ...]
        gaussian   = _gaussian(gray_np)[None, ...]
        sobel_d    = _sobel_magnitude(gray_np)[None, ...]
        normal_var = _normal_variation(gray_np)[None, ...]
        fourier_m  = _fourier(gray_np)[None, ...]

        # HR‑8 通道
        hr = np.concatenate([gray_3,
                             albedo,
                             gaussian,
                             sobel_d,
                             normal_var,
                             fourier_m], axis=0)   # [8,H,W]

        # ---- 低分辨率 downsample ----
        lr_h = self.hr_size // self.r
        lr_w = lr_h
        lr = cv.resize(gray_np, (lr_w, lr_h), interpolation=cv.INTER_AREA)
        lr_3 = np.stack([lr]*3, axis=0)            # [3,lr_h,lr_w]

        # ---- 转 Tensor ----
        lr_t  = torch.from_numpy(lr_3).float()
        hr_t  = torch.from_numpy(hr ).float()
        f_t   = torch.from_numpy(fourier_m).float()

        return lr_t, hr_t, f_t

# ──────────────────────────────────────────────────────────────────
# ----------------  Lightning DataModule  -------------------------
# ──────────────────────────────────────────────────────────────────
class NanoDataLoader(L.LightningDataModule):
    """
    Parameters
    ----------
    root_dir : str
        目录，包含所有图像文件 (tif/png/jpg/exr)。
    r : int
        下采样比例 (scale factor)。
    high_res : int, default=256
        HR patch 边长。
    batch_size : int
        批大小。
    num_workers : int
        DataLoader workers.
    seed : int
        随机划分 seed。
    """

    SUPPORTED_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr")

    def __init__(self,
                 root_dir: str,
                 r: int,
                 high_res: int = 256,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters()

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"{root_dir} 不存在")

        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(self.SUPPORTED_EXT)
        ]
        if not self.image_paths:
            raise FileNotFoundError("未找到任何支持的图像文件")

        # 数据集对象将在 setup() 创建
        self.train_ds = self.val_ds = self.test_ds = None

    # ──────────────────────────────────────────
    def setup(self, stage: Optional[str] = None):
        # 按 8:1:1 划分
        generator = torch.Generator().manual_seed(self.hparams.seed)
        full_ds = _NanoDataset(self.image_paths,
                               self.hparams.r,
                               self.hparams.high_res)
        n_total = len(full_ds)
        n_train = int(0.8 * n_total)
        n_val   = int(0.1 * n_total)
        n_test  = n_total - n_train - n_val

        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_ds, [n_train, n_val, n_test], generator=generator)

    # ──────────────────────────────────────────
    def _make_loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = self.hparams.batch_size,
            shuffle     = shuffle,
            num_workers = self.hparams.num_workers,
            pin_memory  = True,
        )

    def train_dataloader(self): return self._make_loader(self.train_ds, True)
    def val_dataloader  (self): return self._make_loader(self.val_ds,   False)
    def test_dataloader (self): return self._make_loader(self.test_ds,  False)
