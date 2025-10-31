import lmdb
import cv2
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset

class Dataset_(Dataset):
    def __init__(self, lmdb_file, transform=None, train_portion=1.0, shuffle=False, val=False, seed=1):
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.train_portion = train_portion
        self.shuffle = shuffle
        self.val = val
        self.seed = seed

        with lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                self.keys = [key for key, _ in txn.cursor()]

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.keys)

        split_idx = int(len(self.keys) * self.train_portion)
        self.keys = self.keys[split_idx:] if self.val and self.train_portion < 1 else self.keys[:split_idx]

        self.env = None

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_file,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_env()
        key = self.keys[idx]
        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB {self.lmdb_file}")

            image = np.frombuffer(value, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Corrupt image for key {key}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        if self.transform and not self.val:
            image1 = self.transform(image)
            image2 = self.transform(image)
            return image1, image2

        if self.transform:
            image = self.transform(image)
        return image
