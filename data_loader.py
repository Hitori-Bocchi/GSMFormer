# Data loader.
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

class BuildingDataset(Dataset):

    def __init__(self, img_dir, mask_dir, boundary_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.boundary_dir = boundary_dir

        self.images = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        assert len(self.images) == len(self.masks), "Image and mask count mismatch!"

        self.boundaries = [None] * len(self.images)

        self.num_classes = 2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        mask_np = np.array(mask, dtype=np.uint8)
        mask_np[mask_np > 0] = 1
        seg_mask = torch.from_numpy(mask_np).long()

        if idx < len(self.boundaries):
            bname = self.boundaries[idx]
        else:
            bname = None
        boundary_mask = self._make_boundary_from_mask(mask_np).float()

        return image, seg_mask, boundary_mask

    @staticmethod
    def _make_boundary_from_mask(mask_np):

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_np, kernel, iterations=3)
        boundary = (mask_np - eroded)
        boundary[boundary < 0] = 0
        return torch.from_numpy(boundary).byte()
