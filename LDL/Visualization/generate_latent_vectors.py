import os
import time
import random
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from SimCLR_args import conf
from SimCLR import *

if __name__ == "__main__":
    # === Arguments and Setup ===
    args = conf()
    print(args)

    if args.lmdb:
        from DataLoaderLMDB import Dataset_
    else:
        from torchvision.datasets import ImageFolder

    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Transform (just one deterministic view)
    _transform = transforms.Compose([
        transforms.Resize((args.r_crop_size, args.r_crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # === Dataset and DataLoader ===
    if args.lmdb:
        _dataset = Dataset_(
            args.path2traindb,
            transform=_transform,
            train_portion=1.0,
            shuffle=True,
            val=True,  # single-view output
            seed=args.seed
        )
    else:
        _dataset = ImageFolder(args.path2traindb, transform=_transform)

    _loader = DataLoader(
        _dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # === Load SimCLR Model ===
    SimCLR = Network(args)
    if args.resume:
        assert os.path.isfile(args.resume), f"=> no checkpoint found at '{args.resume}'"
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        SimCLR.load_state_dict(new_state_dict)
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint.get('epoch', '?')})")
    else:
        print("Model was not specified in SimCLR 'resume'")
        exit()

    if not args.no_multigpu and torch.cuda.device_count() > 1:
        SimCLR = nn.DataParallel(SimCLR)
        print(f"Using {torch.cuda.device_count()} GPUs")

    SimCLR = SimCLR.to(device)
    SimCLR.eval()

    # === Generate Latent Vectors ===
    with h5py.File('latent_vectors.h5', 'a') as f:
        first_batch = True

        for batch in tqdm(_loader, desc="Computing latent vectors"):
            # Handle single image or tuple of two views
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            imgs = imgs.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                latent_vecs_batch = SimCLR(imgs)
                latent_vecs_batch = latent_vecs_batch.detach().cpu().view(latent_vecs_batch.size(0), -1).numpy()

            # Create or extend dataset
            if first_batch:
                latent_vectors_dataset = f.create_dataset(
                    'latent_vectors',
                    shape=(0, latent_vecs_batch.shape[1]),
                    maxshape=(None, latent_vecs_batch.shape[1]),
                    chunks=True
                )
                first_batch = False
            else:
                latent_vectors_dataset = f['latent_vectors']

            # Resize and append
            latent_vectors_dataset.resize(latent_vectors_dataset.shape[0] + latent_vecs_batch.shape[0], axis=0)
            latent_vectors_dataset[-latent_vecs_batch.shape[0]:] = latent_vecs_batch

    print(" Latent vectors successfully saved to latent_vectors.h5")
