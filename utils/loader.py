# loader.py
这是我之前的数据集，请你阅读之：
import os
import torch
import torch.utils.data as data
import cv2 as cv
import numpy as np
import random

# ====================
# Configuration area 
# pass
# ====================
DEFAULT_PATCH_SIZE = 256  # for training crop
DEFAULT_SEQ_LENGTH = 1  # no temporal data
DEFAULT_SCALE_LIST = [2,4,6,8,10]  # supports 2x, 4x super-resolution
DEFAULT_RANDOM_MASK = False

# ====================
# Dataset Class
# ====================
class NanoMaterialDataset(data.Dataset):
    def __init__(self, paths, scale_list=DEFAULT_SCALE_LIST, lr_mode='downsample', 
                 is_test=False, is_patch=True, patch_size=DEFAULT_PATCH_SIZE):
        super(NanoMaterialDataset, self).__init__()

        self.paths = paths
        self.scale_list = scale_list
        self.lr_mode = lr_mode
        self.is_test = is_test
        self.is_patch = is_patch
        self.patch_size = patch_size

        # Collect all image paths
        self.image_paths = []
        for p in self.paths:
            imgs = sorted([os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr'))])
            self.image_paths.extend(imgs)

    def get_scale(self):
        if self.is_test:
            return self.scale_list[0]
        else:
            return random.choice(self.scale_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        gt_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        if gt_img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        if gt_img.ndim == 2:
            gt_img = gt_img[..., None]
        if gt_img.shape[-1] == 1:
            gt_img = np.repeat(gt_img, 3, axis=-1)  # Ensure 3 channels

        gt_img = gt_img.astype(np.float32) / 255.0
        th, tw, _ = gt_img.shape

        scale = self.get_scale()

        # Prepare LR image
        if self.lr_mode == 'downsample':
            ih, iw = int(th / scale), int(tw / scale)
            lr_img = cv.resize(gt_img, (iw, ih), interpolation=cv.INTER_AREA)
        else:
            raise NotImplementedError("Currently only support 'downsample' mode")

        # Training: Random Crop
        if not self.is_test and self.is_patch:
            ih, iw, _ = lr_img.shape
            lr_crop_size = int(self.patch_size / scale)

            sih = random.randint(0, ih - lr_crop_size)
            siw = random.randint(0, iw - lr_crop_size)

            sth = int(sih * scale)
            stw = int(siw * scale)

            lr_img = lr_img[sih:sih + lr_crop_size, siw:siw + lr_crop_size, :]
            gt_img = gt_img[sth:sth + self.patch_size, stw:stw + self.patch_size, :]

        # Final tensor conversion
        lr_img = torch.from_numpy(lr_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()

        return {
            'lr': lr_img,            # low-resolution image
            'gt': gt_img,            # high-resolution ground-truth
            'scale': float(scale),   # scale factor
            'path': img_path         # original file path
        }
