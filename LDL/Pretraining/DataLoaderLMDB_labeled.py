import lmdb
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import h5py

class Dataset_labeled(Dataset):
    def __init__(self, lmdb_file, path2labels, transform=None, train_portion=1, shuffle=False, val=False, seed=1):
        self.transform = transform
        self.lmdb_file = lmdb_file
        self.path2labels = path2labels

        # Initialize keys from LMDB
        self.keys = []
        with lmdb.open(self.lmdb_file, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    if key_str.startswith('image_'):  
                        self.keys.append(key)
                    
        self.indices = np.arange(len(self.keys))  # Use numpy array for shuffling
        if shuffle:
            np.random.seed(seed)
            self.indices = np.random.permutation(self.indices)
        if val and train_portion < 1:
            self.indices = self.indices[int(len(self.indices) * train_portion):]
        else:
            self.indices = self.indices[:int(len(self.indices) * train_portion)]

        # The HDF5 file is not opened here; it will be opened in __getitem__
        self.env = None

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False)

    def _load_pseudo_labels_h5(self):
        """Open the HDF5 file in each worker."""
        return h5py.File(self.path2labels, 'r')['pseudo_labels']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._init_env()
        image_key = self.keys[self.indices[idx]]
        with self.env.begin() as txn:
            # Retrieve the image from LMDB
            image_value = txn.get(image_key)
            image = np.frombuffer(image_value, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = Image.fromarray(image)

            # Open the HDF5 file for pseudo-labels in __getitem__ to avoid multiprocessing issues
            with h5py.File(self.path2labels, 'r') as f:
                pseudo_label = f['pseudo_labels'][self.indices[idx]]

        if self.transform:
            image = self.transform(image)

        return image, pseudo_label
