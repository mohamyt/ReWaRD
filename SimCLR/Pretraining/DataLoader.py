import os
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, IMG_EXTENSIONS


class Dataset_(Dataset):
    """Dataset that loads images using torchvision's default loader and
    sets classes according to subfolders (like ImageFolder).

    Return format is kept compatible with the previous implementation:
    - For training (val=False): returns (image1, image2) where both are transformed tensors.
    - For validation (val=True): returns a single transformed image.

    Attributes added for compatibility with ImageFolder: `classes`, `class_to_idx`, `samples`.
    """

    def __init__(self, image_dir, transform=None, train_portion=1.0, shuffle=False, val=False, seed=1):
        self.image_dir = image_dir
        self.transform = transform
        self.train_portion = train_portion
        self.shuffle = shuffle
        self.val = val
        self.seed = seed

        # Discover classes from subdirectories (like ImageFolder)
        classes = [d for d in sorted(os.listdir(image_dir)) if os.path.isdir(os.path.join(image_dir, d))]
        if classes:
            self.classes = classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            samples = []
            for cls_name in self.classes:
                cls_idx = self.class_to_idx[cls_name]
                cls_folder = os.path.join(image_dir, cls_name)
                for root, _, fnames in os.walk(cls_folder):
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            samples.append((path, cls_idx))
        else:
            # No subfolders: treat all images in root as one class with idx 0
            self.classes = []
            self.class_to_idx = {}
            samples = []
            for fname in sorted(os.listdir(image_dir)):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    samples.append((os.path.join(image_dir, fname), 0))

        if not samples:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Keep samples and a separate list of file paths for compatibility
        self.samples = samples
        self.image_files = [s[0] for s in samples]
        self.labels = [s[1] for s in samples]

        if self.shuffle:
            random.seed(self.seed)
            combined = list(zip(self.image_files, self.labels))
            random.shuffle(combined)
            self.image_files, self.labels = zip(*combined)
            self.image_files = list(self.image_files)
            self.labels = list(self.labels)

        split_idx = int(len(self.image_files) * self.train_portion)
        if self.val and self.train_portion < 1:
            self.image_files = self.image_files[split_idx:]
            self.labels = self.labels[split_idx:]
        else:
            self.image_files = self.image_files[:split_idx] if self.train_portion < 1 else self.image_files
            self.labels = self.labels[:split_idx] if self.train_portion < 1 else self.labels


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        label = self.labels[idx] if self.labels is not None and len(self.labels) > idx else None

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Apply transforms
        if self.transform and not self.val:
            image1 = self.transform(img)
            image2 = self.transform(img)
            return image1, image2

        if self.transform:
            img = self.transform(img)

        return img
