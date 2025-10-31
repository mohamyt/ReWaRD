import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Dataset_(Dataset):
    def __init__(self, image_dir, transform=None, train_portion=1.0, shuffle=False, val=False, seed=1):
        self.image_dir = image_dir
        self.transform = transform
        self.train_portion = train_portion
        self.shuffle = shuffle
        self.val = val
        self.seed = seed

        self.image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ]

        if not self.image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.image_files)

        split_idx = int(len(self.image_files) * self.train_portion)
        self.image_files = self.image_files[split_idx:] if self.val and self.train_portion < 1 else self.image_files[:split_idx]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform and not self.val:
            image1 = self.transform(image)
            image2 = self.transform(image)
            return image1, image2

        if self.transform:
            img = self.transform(img)
        return img
