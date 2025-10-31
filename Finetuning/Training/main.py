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
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import sys
import importlib



def loadmodel(args):
    # Load configuration options
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
    

    class AutoencoderModel(nn.Module):
        def __init__(self, original_model, num_classes, device=device):
            super(AutoencoderModel, self).__init__()
            self.original_model = original_model
            self.pool = nn.AdaptiveAvgPool2d((1,1))

            # determine num_features dynamically
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(device)
                feat_map = self.original_model(dummy)
                num_features = self.pool(feat_map).view(1, -1).size(1)

            self.classification_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)   # logits
            )

        def forward(self, x):
            x = self.original_model(x)
            x = self.pool(x)
            x = self.classification_head(x)
            return x

    class SimCLRModel(nn.Module):
        def __init__(self, original_model, num_classes):
            super(SimCLRModel, self).__init__()
            self.original_model = original_model
            feat_dim = original_model.get_feature_dim()
            self.classification_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),          # Final classification layer
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
            #self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.fc(x)
            return x

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
        top1_train_acc = checkpoint.get('top1_train_acc', [])
        top5_train_acc = checkpoint.get('top5_train_acc', [])
        top1_val_acc = checkpoint.get('top1_val_acc', [])
        top5_val_acc = checkpoint.get('top5_val_acc', [])
        lr = checkpoint.get('lr', [])
        current_lr = checkpoint.get('current_lr', args.lr)
        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        train_losses = []
        batch_losses = []
        val_batch_losses = []
        val_losses = []
        lr = []
        top1_train_acc, top5_train_acc, top1_val_acc, top5_val_acc = [], [], [], []
        current_lr = args.lr

    # Multi-GPU setup
    if not args.no_multigpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Optimizer and criterion setup
    optimizer = optim.SGD(model.parameters(), lr=args.lr if not args.use_last_lr else current_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

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

    return model, train_loader, val_loader, train_dataset, val_dataset, device, optimizer, criterion, scheduler, scaler, save_checkpoint, weight_folder, lr, train_losses, batch_losses, val_batch_losses, val_losses, starttime, top1_train_acc, top5_train_acc, top1_val_acc, top5_val_acc

def visualize_batch(val_loader, dataset, save_path="./imgs/sample_batch.png"):
    imgs, labels = next(iter(val_loader))  # grab one batch
    imgs = imgs[:16]  # take first 16
    labels = labels[:16]

    # Undo normalization for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    plt.figure(figsize=(12, 12))
    for i in range(len(imgs)):
        img = imgs[i].permute(1, 2, 0).numpy()  # C,H,W → H,W,C
        img = std * img + mean                  # unnormalize
        img = np.clip(img, 0, 1)

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        class_name = dataset.classes[labels[i].item()]
        plt.title(class_name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(f"Sample batch saved to {save_path}")
    plt.show()




def main(model, train_loader, val_loader, train_dataset, val_dataset, device, optimizer, criterion, scheduler, scaler, save_checkpoint, weight_folder, lr, train_losses, batch_losses, val_batch_losses, val_losses, starttime, top1_train_acc, top5_train_acc, top1_val_acc, top5_val_acc):
    # Training loop

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_top1, correct_top5, total = 0, 0, 0
        lr.append(optimizer.param_groups[0]["lr"])
        print(f"Learning rate: {lr[-1]}")

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, labels = imgs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            batch_losses.append(loss.item() * imgs.size(0))

            _, pred = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            correct_top1 += (pred[:, 0] == labels).sum().item()
            correct_top5 += sum([labels[i] in pred[i] for i in range(labels.size(0))])

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        top1_train_acc.append(correct_top1 / total)
        top5_train_acc.append(correct_top5 / total)


        if args.val:
            model.eval()
            running_loss = 0.0
            correct_top1, correct_top5, total = 0, 0, 0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Validation of epoch {epoch+1}/{args.epochs}"):
                    imgs, labels = imgs.to(device).float(), labels.to(device).long()

                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)

                    running_loss += loss.item() * imgs.size(0)
                    val_batch_losses.append(loss.item() * imgs.size(0))

                    _, pred = outputs.topk(5, 1, True, True)
                    total += labels.size(0)
                    correct_top1 += (pred[:, 0] == labels).sum().item()
                    correct_top5 += sum([labels[i] in pred[i] for i in range(labels.size(0))])

            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_loss)
            top1_val_acc.append(correct_top1 / total)
            top5_val_acc.append(correct_top5 / total)

        scheduler.step()

        # --- Checkpoint saving logic ---
        endtime = time.time()
        interval = endtime - starttime
        avg_epoch_time = interval / (epoch + 1)
        time_limit = 24 * 3600 #time limit on tinyx
        time_left = time_limit - interval
        save_due_to_time = time_left < 2 * avg_epoch_time

        save_this_epoch = (epoch + 1) % args.save_interval == 0 or save_due_to_time
        if save_this_epoch:
            dir_path = f"./data/weight/{args.usenet}/{args.dataset}/{args.model_type}{weight_folder}"
            checkpoint_filename = f"{dir_path}/checkpoint_{args.model_type}_epoch_{epoch+args.start_epoch}_{args.usenet}_{args.dataset}.pth.tar"
            if not os.path.exists(dir_path):
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
                'current_lr': optimizer.param_groups[0]["lr"],
                'top1_train_acc': top1_train_acc,
                'top5_train_acc': top5_train_acc,
                'top1_val_acc': top1_val_acc,
                'top5_val_acc': top5_val_acc
            }, checkpoint_filename)

            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {train_losses[-1]}")

            # Clear CUDA memory cache
            torch.cuda.empty_cache()

        print(f"Epoch [{epoch+1}/{args.epochs}] complete. Training Loss: {train_losses[-1]:.4f}, Top-1 Acc: {top1_train_acc[-1]:.4f}, Top-5 Acc: {top5_train_acc[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Top-1 Acc: {top1_val_acc[-1]:.4f}, Top-5 Acc: {top5_val_acc[-1]:.4f}")
        # Final processing time
        endtime = time.time()
        interval = endtime - starttime
        print(f"Elapsed time: {int(interval // 3600)}h {int((interval % 3600) // 60)}m {int(interval % 60)}s")


if __name__ == "__main__":
    args_module_name = "args"
    if len(sys.argv) > 1:
        args_module_name = sys.argv[1].replace(".py", "")

    args_module = importlib.import_module(args_module_name)
    args = args_module.conf()

    print("Configuring Model...")
    model, train_loader, val_loader, train_dataset, val_dataset, device, optimizer, criterion, scheduler, scaler, save_checkpoint, weight_folder, lr, train_losses, batch_losses, val_batch_losses, val_losses, starttime, top1_train_acc, top5_train_acc, top1_val_acc, top5_val_acc = loadmodel(args)

    if val_loader is not None and val_dataset is not None:
        print("Visualizing sample batch...")
        visualize_batch(val_loader, val_dataset, save_path="./imgs/" + args.model_type + weight_folder + "/sample_batch.png")

    print("Starting training...")
    main(model, train_loader, val_loader, train_dataset, val_dataset, device, optimizer, criterion, scheduler, scaler, save_checkpoint, weight_folder, lr, train_losses, batch_losses, val_batch_losses, val_losses, starttime, top1_train_acc, top5_train_acc, top1_val_acc, top5_val_acc)
