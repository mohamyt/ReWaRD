import os
import random
import time
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from args import conf
from transforms import RandomGaussianBlur
from Model import Network

# debug imports
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def get_num_features(model, input_shape=(3, 224, 224)):
    """Pass a dummy input through the backbone to auto-detect output size."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape).to(next(model.parameters()).device)
        out = model.backbone(dummy)
        return out.shape[1] if len(out.shape) == 2 else out.numel()


if __name__ == "__main__":
    args = conf()
    run_time = 24 * 3600
    print(f"Config:\n{args}")

    if args.lmdb:
        from DataLoaderLMDB import Dataset_
        from DataLoaderLMDB_labeled import Dataset_labeled
    else:
        from DataLoader import Dataset_

    starttime = time.time()
    today = datetime.now()
    weight_folder = "/" + today.strftime('%Y%m%d_%H%M%S')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
        RandomGaussianBlur(probability=args.p_blur, min_radius=args.min_blur_r, max_radius=args.max_blur_r),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_portion = 1.0
    train_dataset = Dataset_labeled(
        args.path2traindb,
        transform=train_transform,
        train_portion=train_portion,
        shuffle=False,
        val=False,
        seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2,
    )

    # debugging: show 6 images & pseudo-labels
    try:
        # grab one batch
        batch = next(iter(train_loader))
        imgs_batch, labels_batch = batch  # expected (imgs, pseudo_labels) from Dataset_labeled

        # make sure we have tensors
        imgs_batch = imgs_batch.detach().cpu()
        labels_batch = labels_batch.detach().cpu()

        # Print label shape
        print(f"DEBUG: pseudo labels shape = {labels_batch.shape} (dtype={labels_batch.dtype})")

        # Decide how many to show
        show_n = min(6, imgs_batch.shape[0])
        imgs_vis = imgs_batch[:show_n]
        labels_vis = labels_batch[:show_n]

        # Denormalize images for plotting
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def denormalize_tensor(img_t):
            # img_t: C,H,W tensor in [normalized]
            arr = img_t.numpy()
            arr = np.transpose(arr, (1, 2, 0))  # H,W,C
            arr = (arr * std) + mean
            arr = np.clip(arr, 0.0, 1.0)
            return arr

        # Prepare figure
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i in range(show_n):
            ax = axes[i]
            img_np = denormalize_tensor(imgs_vis[i])
            ax.imshow(img_np)
            ax.axis('off')

            # Prepare label text
            if labels_vis.ndim == 2 and labels_vis.dtype.is_floating_point:
                # soft pseudo-labels: show top-3 indices and probs
                probs = labels_vis[i].numpy()
                if probs.sum() == 0:
                    label_text = "all-zero pseudo-label!"
                else:
                    topk = 3
                    topk_idx = np.argsort(-probs)[:topk]
                    topk_vals = probs[topk_idx]
                    pairs = [f"{idx}:{val:.2f}" for idx, val in zip(topk_idx, topk_vals)]
                    label_text = "soft top3: " + ", ".join(pairs)
            else:
                # hard labels (integers)
                if labels_vis.ndim == 1:
                    label_text = f"label: {int(labels_vis[i].item())}"
                elif labels_vis.ndim == 2:
                    # one-hot or multi-hot
                    idx = int(labels_vis[i].argmax().item())
                    label_text = f"hard(onehot) label: {idx}"
                else:
                    label_text = f"labels shape: {labels_vis[i].shape}"

            ax.set_title(label_text, fontsize=9)

        for j in range(show_n, 6):
            axes[j].axis('off')

        plt.suptitle("Sample batch (first up to 6) and their pseudo labels")
        os.makedirs("./imgs", exist_ok=True)
        savepath = "./imgs/pseudo_label_batch.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(savepath, dpi=150)
        plt.show()
        print(f"DEBUG: Saved sample figure to {savepath}")

    except Exception as e:
        print("DEBUG: Failed to create pseudo-label visualization:", e)
    # end debug block

    # validation loader
    if args.val:
        val_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.RandomResizedCrop((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_dataset = Dataset_labeled(
            args.path2valdb,
            transform=val_transform,
            train_portion=0.8,
            shuffle=False,
            val=True,
            seed=args.seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2,
        )

    # initialize model
    model = Network(args).to(device)

    # fix feature size
    try:
        num_features = get_num_features(model)
        if model.fc.in_features != num_features:
            model.fc = nn.Linear(num_features, args.numof_classes).to(device)
            print(f"Fixed model.fc input dim to {num_features}")
    except Exception as e:
        print(f"⚠️ Could not auto-detect feature size: {e}")

    # multi-gpu
    if not args.no_multigpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

    # === Optimizer, criterion, scheduler ===
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # For classification with logits, CrossEntropyLoss is appropriate for hard labels.
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # resume checkpoint
    train_losses, val_losses, lr_log = [], [], []
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        lr_log = checkpoint.get('lr', [])
        train_acc1 = checkpoint.get('top1_train_acc', [])
        train_acc5 = checkpoint.get('top5_train_acc', [])
        val_acc1 = checkpoint.get('top1_val_acc', [])
        val_acc5 = checkpoint.get('top5_val_acc', [])
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        train_losses = []
        batch_losses = []
        val_batch_losses = []
        val_losses = []
        lr = []
        top1_train_acc, top5_train_acc, top1_val_acc, top5_val_acc = [], [], [], []
        current_lr = args.lr

    # training loop
    print("Starting training...")
    epoch_times = []  # track duration per epoch (train+val)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss, total, correct_top1, correct_top5 = 0.0, 0, 0, 0
        lr_log.append(optimizer.param_groups[0]["lr"])
        print(f"\nEpoch [{epoch+1}/{args.epochs}] | LR: {lr_log[-1]:.6f}")

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Use non_blocking transfer when pin_memory=True to overlap host->device copies
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)

                # Handle soft pseudo-labels
                if labels.dtype == torch.float32 and labels.ndim == 2:
                    loss = -(labels * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                else:
                    loss = criterion(outputs, labels.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)

            # Top-1/5 accuracy
            _, preds = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            # Top-1
            correct_top1 += (preds[:, 0] == labels.long()).sum().item()
            # Top-5: handle hard integer labels or soft pseudo-labels (float vectors)
            if labels.dtype == torch.float32 and labels.ndim == 2:
                # soft labels -> get predicted class by argmax of the soft vector
                lbl = labels.argmax(dim=1).long()
            else:
                lbl = labels.long()
            # preds: (batch, 5); compare and count any match per sample
            matches = preds.eq(lbl.view(-1, 1))
            correct_top5 += matches.any(dim=1).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc1 = correct_top1 / total
        train_acc5 = correct_top5 / total
        train_losses.append(epoch_loss)

        print(f"Loss: {epoch_loss:.4f} | Top-1: {train_acc1:.4f} | Top-5: {train_acc5:.4f}")

        scheduler.step()
        top1_train_acc.append(train_acc1)
        top5_train_acc.append(train_acc5)

        # === Validation ===
        if args.val:
            model.eval()
            running_loss = 0.0
            correct_top1, correct_top5, total = 0, 0, 0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validation"):
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        if labels.dtype == torch.float32 and labels.ndim == 2:
                            loss = -(labels * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                        else:
                            loss = criterion(outputs, labels.long())
                    running_loss += loss.item() * imgs.size(0)

                    _, preds = outputs.topk(5, 1, True, True)
                    total += labels.size(0)
                    # Top-1
                    correct_top1 += (preds[:, 0] == labels.long()).sum().item()
                    # Top-5 (handle soft/hard labels)
                    if labels.dtype == torch.float32 and labels.ndim == 2:
                        lbl = labels.argmax(dim=1).long()
                    else:
                        lbl = labels.long()
                    matches = preds.eq(lbl.view(-1, 1))
                    correct_top5 += matches.any(dim=1).sum().item()
            val_loss = running_loss / len(val_loader.dataset)
            val_acc1 = correct_top1 / total
            val_acc5 = correct_top5 / total
            val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f} | Top-1: {val_acc1:.4f} | Top-5: {val_acc5:.4f}")
        else:
            val_loss = None
            val_acc1 = 0.0
            val_acc5 = 0.0

        epoch_dur = time.time() - epoch_start
        epoch_times.append(epoch_dur)

        top1_val_acc.append(val_acc1)
        top5_val_acc.append(val_acc5)
        # === Save checkpoint ===
        should_save = False
        if (epoch + 1) % args.save_interval == 0:
            should_save = True

        if len(epoch_times) > 0:
            mean_epoch = float(np.mean(epoch_times))
            elapsed = time.time() - starttime
            time_left = run_time - elapsed
            # save if time_left_to_24h is positive and less than 1.5x mean epoch time
            if time_left > 0 and time_left < 1.5 * mean_epoch:
                print(f"Saving checkpoint because time left to 24h ({time_left:.1f}s) < 1.5*mean_epoch ({1.5*mean_epoch:.1f}s)")
                should_save = True

        if should_save:
            dir_path = f"./data/weight/{args.usenet}/{args.dataset}{weight_folder}"
            os.makedirs(dir_path, exist_ok=True)
            checkpoint_path = f"{dir_path}/epoch_{epoch+1}.pth.tar"
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'lr': lr_log,
                'current_lr': optimizer.param_groups[0]["lr"],
                'current_lr': optimizer.param_groups[0]["lr"],
                'top1_train_acc': top1_train_acc,
                'top5_train_acc': top5_train_acc,
                'top1_val_acc': top1_val_acc,
                'top5_val_acc': top5_val_acc
            }, checkpoint_path)

        torch.cuda.empty_cache()

    # wrap up
    total_time = time.time() - starttime
    print(f"\nTraining completed in {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s.")
