import lmdb
import cv2
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset

class Dataset_(Dataset):
    def __init__(self, lmdb_file, transform=None, train_portion=1, shuffle=False ,val=False, seed=1):
        self.transform = transform
        self.lmdb_file = lmdb_file

        self.keys = []
        with lmdb.open(self.lmdb_file, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    if key_str.startswith('image_'):  
                        self.keys.append(key)
        if shuffle:
            random.seed(seed)
            random.shuffle(self.keys)
        if val and train_portion<1:
            self.keys = self.keys[int(len(self.keys)*train_portion):]
        else:
            self.keys = self.keys[:int(len(self.keys)*train_portion)]
        # Class-level cache for the LMDB environment
        self.env = None

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_env()
        image_key = self.keys[idx]
        with self.env.begin() as txn:
            # Retrieve the image
            image_value = txn.get(image_key)
            image = np.frombuffer(image_value, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = Image.fromarray(image)
            
            # Retrieve the class label using a corresponding class key
            class_key = image_key.decode('utf-8').replace('image_', 'class_').encode('utf-8')
            class_value = txn.get(class_key)
            class_label = class_value.decode('utf-8')  
            class_label = int(class_label)
        if self.transform:
            image = self.transform(image)

        return image, class_label  
