import os
import random
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import sys
import importlib

from transforms import RandomGaussianBlur
from Model import *
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt
import random

def sanity_check_dataloader(train_loader, num_samples=8, device="cpu", seed=1):
    dataset = train_loader.dataset

    # Select random indices based on seed
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)

    imgs1, imgs2 = [], []
    for idx in indices:
        img_pair = dataset[idx]
        # Unpack correctly depending on return type
        if isinstance(img_pair, (tuple, list)) and len(img_pair) == 2:
            img1, img2 = img_pair
        else:
            img1, img2 = img_pair, img_pair  # fallback: same image twice
        imgs1.append(img1)
        imgs2.append(img2)

    # Stack to tensors
    imgs1 = torch.stack(imgs1).to(device)
    imgs2 = torch.stack(imgs2).to(device)

    # Denormalize for visualization
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    imgs1 = denormalize(imgs1, mean, std).clamp(0, 1)
    imgs2 = denormalize(imgs2, mean, std).clamp(0, 1)

    # Plot the pairs
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    for i in range(num_samples):
        ax1, ax2 = axes[i]
        ax1.imshow(imgs1[i].permute(1, 2, 0).cpu().numpy())
        ax2.imshow(imgs2[i].permute(1, 2, 0).cpu().numpy())
        ax1.axis("off")
        ax2.axis("off")
        ax1.set_title(f"View 1 - sample {i}")
        ax2.set_title(f"View 2 - sample {i}")

    plt.tight_layout()
    plt.show()
    plt.savefig("sample_images.png")

if __name__ == "__main__":
    # Default args module
    args_module_name = "args"

    if len(sys.argv) > 1:
        args_path = sys.argv[1]
        args_module_name = os.path.splitext(os.path.basename(args_path))[0]

    print(f"Using argument file: {args_module_name}.py")

    args_module = importlib.import_module(args_module_name)
    args = args_module.conf()

    # Option
    args = conf()
    print(args)

    if args.lmdb:
        from DataLoaderLMDB import Dataset_
    else:
        from DataLoader import Dataset_

    # Processing time
    starttime = time.time()
    today = datetime.now()
    weight_folder = "/" + today.strftime('%Y%m%d') + str(today.hour) + str(today.minute)

    # GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # to deterministic
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training settings

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.r_crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=args.p_grayscale),
        RandomGaussianBlur(
            probability=args.p_blur,
            radius=torch.rand(1).item() * (args.max_blur_r - args.min_blur_r) + args.min_blur_r
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    train_portion = 1

    if args.val:
        val_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=args.r_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if args.path2valdb == args.path2traindb:
            train_portion = 0.7
        if args.lmdb:
            val_dataset = Dataset_(args.path2valdb, transform=val_transform, train_portion=train_portion, shuffle=True, val=True, seed=args.seed)
        else:
            val_dataset = Dataset_(args.path2valdb, transform=val_transform, train_portion=train_portion, shuffle=True, val=True, seed=args.seed)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    if args.lmdb:
        train_dataset = Dataset_(args.path2traindb, transform=train_transform, train_portion=train_portion, shuffle=True, val=False, seed=args.seed)
    else:
        train_dataset = Dataset_(args.path2traindb, transform=val_transform, train_portion=train_portion, shuffle=True, val=False, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # NT-Xent loss function
    def nt_xent_loss(projections, temperature):
        batch_size = projections.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(projections.device)

        # Normalize projections
        projections = F.normalize(projections, p=2, dim=1)
        similarity_matrix = torch.matmul(projections, projections.T)

        # Mask to remove self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(projections.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long).to(projections.device)

        return F.cross_entropy(logits, targets)

    # Model
    model = Network(args)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        args.start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        batch_losses = checkpoint.get('batch_losses', [])
        val_batch_losses = checkpoint.get('val_batch_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        lr = checkpoint.get('lr', [])
        current_lr = checkpoint.get('current_lr', [])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        print("Last used learning rate: {}".format(lr[-1]))

    if not args.no_multigpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    if 'train_losses' not in locals():
        train_losses = []
        batch_losses = []
        val_batch_losses = []
        val_losses = []
        lr = []
        current_lr = args.lr

    # Using AdamW optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr if not args.use_last_lr else current_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
        if args.resume_scheduler and args.resume:
            for i in range(args.start_epoch):
                scheduler.step()

    num_epochs = args.epochs
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    def save_checkpoint(state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    sanity_check_dataloader(train_loader, device=device, seed=int(starttime))
    #Train loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        lr.append(optimizer.param_groups[0]["lr"])
        print(f"Learning rate: {lr[-1]}")
        
        for imgs1, imgs2 in tqdm(iterable=train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # if not args.lmdb:
            #     imgs1, imgs2 = imgs1[0], imgs2[0]
            imgs = torch.cat([imgs1, imgs2], dim=0)
            #print(f"imgs shape: {imgs.shape}")
            imgs = imgs.to(device)

            # Check for NaNs in input data
            if torch.isnan(imgs).any():
                print("NaN detected in input images, skipping this batch.")
                continue

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = nt_xent_loss(outputs, args.temperature)

            if torch.isnan(loss):
                print("NaN detected in loss, skipping this batch.")
                continue

            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            batch_losses.append(loss.item() * imgs.size(0))

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if args.val:
            model.eval()
            running_loss = 0.0
            for imgs in tqdm(iterable=val_loader, desc=f"Validation of epoch {epoch+1}/{num_epochs}"):
                # if not args.lmdb:
                #     imgs = imgs[0]
                imgs = torch.cat([imgs, imgs], dim=0)
                imgs = imgs.to(device)

                if torch.isnan(imgs).any():
                    print("NaN detected in validation images, skipping this batch.")
                    continue

                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = nt_xent_loss(outputs, args.temperature)

                if torch.isnan(loss):
                    print("NaN detected in validation loss, skipping this batch.")
                    continue

                running_loss += loss.item() * imgs.size(0)
                val_batch_losses.append(loss.item() * imgs.size(0))

            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_loss)
        if args.use_scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(f"./data/weight/{args.usenet}/{args.dataset}" + weight_folder, exist_ok=True)
            checkpoint_filename = f"./data/weight/{args.usenet}/{args.dataset}{weight_folder}/checkpoint_{args.model_type}_epoch_{epoch+args.start_epoch}_{args.usenet}_{args.dataset}.pth.tar"
            save_checkpoint({
                'epoch': epoch + args.start_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'batch_losses': batch_losses,
                'val_batch_losses': val_batch_losses,
                'val_losses': val_losses,
                'lr': lr,
                'current_lr': current_lr
            }, checkpoint_filename)
            print(f"Directory '{checkpoint_filename}' created successfully.")
            print(f"Model checkpoint saved at epoch {epoch+1}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]}")

        endtime = time.time()
        interval = endtime - starttime
        print("Elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
        torch.cuda.empty_cache()

    print("Training completed.")
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
