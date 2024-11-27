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

from args import conf
from transforms import RandomGaussianBlur
from Autoencoder import *


if __name__ == "__main__":
    # Load config
    args = conf()
    print(args)

    if args.lmdb:
        from DataLoaderLMDB import Dataset_
    else:
        from torchvision.datasets import ImageFolder
    
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

    if args.dataset == 'ImageNet1k':
        train_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
            RandomGaussianBlur(probability=0.3, radius=2),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Data Normalized with ImageNet1k mean and std.")
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
            RandomGaussianBlur(probability=0.3, radius=2),
            transforms.ToTensor()
        ])
        print("Data not normalized.")

    train_portion = 1

    if args.val:
        if args.dataset == 'ImageNet1k':
            val_transform = transforms.Compose([
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            val_transform = transforms.Compose([
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
                transforms.ToTensor()
            ])
        if args.path2valdb == args.path2traindb:
            train_portion = 0.7
        if args.lmdb:
            val_dataset = Dataset_(args.path2valdb, transform=val_transform, train_portion=train_portion, shuffle=True ,val=True, seed=args.seed)
        else:
            val_dataset = ImageFolder(args.path2valdb, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.lmdb:
        train_dataset = Dataset_(args.path2traindb, transform=train_transform, train_portion=train_portion, shuffle=True ,val=False, seed=args.seed)
    else:
        train_dataset = ImageFolder(args.path2traindb, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Model & optimizer
    model = Network(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)
    if args.resume_scheduler and args.resume:
        for i in range(args.start_epoch):
            scheduler.step()
    
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        # Load the checkpoint
        checkpoint = torch.load(args.resume)

        # Adjust the keys of state_dict if necessary
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v

        # Load state_dict into the model
        model.load_state_dict(new_state_dict)
        args.start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        batch_losses = checkpoint.get('batch_losses', [])
        val_batch_losses = checkpoint.get('val_batch_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    if not args.no_multigpu:
        model = nn.DataParallel(model)

    if not 'train_losses' in locals():  
        train_losses = []
        batch_losses = []
        val_batch_losses = []
        val_losses = []

    num_epochs = args.epochs
    model.to(device)

    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Save checkpoint function
    def save_checkpoint(state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs in tqdm(iterable=train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if not args.lmdb:
                imgs = imgs[0]
            optimizer.zero_grad()
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, imgs)

            scaler.scale(loss).backward()
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
                if not args.lmdb:
                    imgs = imgs[0]
                imgs = imgs.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, imgs)

                running_loss += loss.item() * imgs.size(0)
                val_batch_losses.append(loss.item() * imgs.size(0))

            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_loss)

        scheduler.step()

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
            }, checkpoint_filename)
            print(f"Model checkpoint saved at epoch {epoch+1}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]}")
        # Processing time
        endtime = time.time()
        interval = endtime - starttime
        print("Elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
        # Clear unused memory
        torch.cuda.empty_cache()

    print("Training completed.")
    # Processing time
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
