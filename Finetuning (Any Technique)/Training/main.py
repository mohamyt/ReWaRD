import os
import random
import time
from datetime import datetime
from tqdm import tqdm
import h5py
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
import numpy as np
from torchvision.datasets import ImageFolder



if __name__ == "__main__":
    # Load configuration options
    args = conf()
    print(args)

    # Setup
    starttime = time.time()
    today = datetime.now()
    weight_folder = "/" + today.strftime('%Y%m%d') + str(today.hour) + str(today.minute)

    # CUDA setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset setup
    train_portion = 1
    train_dataset = ImageFolder(args.path2traindb, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # If validation is enabled, define validation dataset and loader
    if args.val:
        val_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if args.path2valdb == args.path2traindb:
            train_portion = 0.8  # Split the training data for validation

        val_dataset = ImageFolder(args.path2valdb, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Initialize model
    def load_model(model_args):
        model_args = conf()
        model = Network(model_args)

        #Loading model
        assert os.path.isfile(args.path2model), f"=> no model found at '{args.path2model}'"
        print(f"=> loading checkpoint '{args.path2model}'")
        return model
    
    if args.model_type=="Autoencoder":
        from Autoencoder import Network, model_select
        from Autoencoder_args import conf
        model_args = conf()
        model = load_model(model_args)
    elif args.model_type=="SimCLR":
        from SimCLR import Network, model_select
        from SimCLR_args import conf
        model_args = conf()
        model = load_model(model_args)
    elif args.model_type=="LDL":
        from LDL import Network, model_select
        from LDL_args import conf
        model_args = conf()
        model = load_model(model_args)
        print("Regular, unpretrained model loaded")

    

    class AutoencoderModel(nn.Module):
        def __init__(self, original_model, num_classes):
            super(AutoencoderModel, self).__init__()
            self.original_model = original_model
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 256),          # Intermediate layer
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),          # Final classification layer
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.original_model(x)
            x = self.classification_head(x)
            return x

    class SimCLRModel(nn.Module):
        def __init__(self, original_model, num_classes):
            super(SimCLRModel, self).__init__()
            self.original_model = original_model
            self.classification_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),          # Intermediate layer
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),          # Final classification layer
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.original_model(x)
            x = self.classification_head(x)
            return x

    class LDLModel(nn.Module):
        def __init__(self, original_model, num_classes):
            super(LDLModel, self).__init__()
            untrained_model = model_select(model_args)
            self.backbone = original_model
            num_labels = num_classes
            num_features = untrained_model.fc.in_features  # Extract features of final layer

            # Fully connected layer to predict label distributions
            self.fc = nn.Linear(num_features, num_labels)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.fc(x)
            return self.softmax(x)

    if args.model_type=="Autoencoder":
        model = model.encoder
        model = AutoencoderModel(model, args.numof_classes)
    elif args.model_type=="SimCLR":
        model = model.encoder
        model = SimCLRModel(model, args.numof_classes)
    elif args.model_type=="LDL":
        model = model.backbone
        model = LDLModel(model, args.numof_classes)
    else:
        from Autoencoder import model_select
        model = model_select(args)

    # Resume from checkpoint if specified
    if args.resume:
        assert os.path.isfile(args.resume), f"=> no checkpoint found at '{args.resume}'"
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        args.start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        batch_losses = checkpoint.get('batch_losses', [])
        val_batch_losses = checkpoint.get('val_batch_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        lr = checkpoint.get('lr', [])
        current_lr = checkpoint.get('current_lr', args.lr)
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        train_losses = []
        batch_losses = []
        val_batch_losses = []
        val_losses = []
        lr = []
        current_lr = args.lr

    # Multi-GPU setup
    if not args.no_multigpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Optimizer and criterion setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr if not args.use_last_lr else current_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

    if args.resume_scheduler and args.resume:
        for i in range(args.start_epoch):
            scheduler.step()

    # Move model to device
    model.to(device)

    # Use AMP for mixed precision training to save memory
    scaler = torch.cuda.amp.GradScaler()

    # Function to save checkpoints
    def save_checkpoint(state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        lr.append(optimizer.param_groups[0]["lr"])
        print(f"Learning rate: {lr[-1]}")

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, labels = imgs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                labels = labels.long()
                loss = criterion(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track loss
            running_loss += loss.item() * imgs.size(0)
            batch_losses.append(loss.item() * imgs.size(0))

        # Calculate and log epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation (if enabled)
        if args.val:
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Validation of epoch {epoch+1}/{args.epochs}"):
                    imgs, labels = imgs.to(device).float(), labels.to(device).float()

                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        labels = labels.long()
                        loss = criterion(outputs, labels)

                    running_loss += loss.item() * imgs.size(0)
                    val_batch_losses.append(loss.item() * imgs.size(0))

            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_loss)

        # Scheduler step
        scheduler.step()

        # Save checkpoints periodically
        if (epoch + 1) % args.save_interval == 0:
            dir_path = f"./data/weight/{args.usenet}/{args.dataset}/{args.model_type}{weight_folder}"
            checkpoint_filename = f"{dir_path}/checkpoint_{args.model_type}_epoch_{epoch+args.start_epoch}_{args.usenet}_{args.dataset}.pth.tar"
            if not os.path.exists(dir_path):
                # Create directory
                os.makedirs(dir_path)
                print(f"Directory '{dir_path}' created successfully.")
            save_checkpoint({
                'epoch': epoch + args.start_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'batch_losses': batch_losses,
                'val_batch_losses': val_batch_losses,
                'val_losses': val_losses,
                'lr': lr,
                'current_lr': optimizer.param_groups[0]["lr"]
            }, checkpoint_filename)

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_losses[-1]}")

        # Clear CUDA memory cache
        torch.cuda.empty_cache()

    print("Training completed.")
    # Final processing time
    endtime = time.time()
    interval = endtime - starttime
    print(f"Elapsed time: {int(interval // 3600)}h {int((interval % 3600) // 60)}m {int(interval % 60)}s")
