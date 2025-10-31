import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from args import conf
from multiprocessing import freeze_support

#This script does the same as main.py but for multiple models in one run, generating comparative plots and a summary table (used for the report).

# ===== Weight names and model types need to be defined for multiple model visualization =====
weight_names = [
    'checkpoint_Autoencoder_epoch_250_resnet18_ImageNet1k.pth.tar',
    'checkpoint_SimCLR_epoch_240_resnet18_ImageNet1k.pth.tar',
    'checkpoint_LDL_epoch_240_resnet18_ImageNet1k.pth.tar',
    'checkpoint_None_epoch_250_resnet18_ImageNet1k.pth.tar'
]

model_types = [
    'Autoencoder',
    'SimCLR',
    'LDL',
    'None'
]

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = conf()
numof_classes = args.numof_classes
max_layer = 1

# Dataset setup (shared)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_dataset = ImageFolder(args.path2valdb, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

# CUDA and seed setup
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.deterministic = True
random.seed(args.seed)
torch.manual_seed(args.seed)

# Storage for summary
summary_data = []

# Accuracy graphs
all_top1_val_acc = {}
all_top5_val_acc = {}

# ===== MODEL WRAPPERS =====
class AutoencoderModel(nn.Module):
    def __init__(self, original_model, num_classes, device=device):
        super().__init__()
        self.original_model = original_model
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            feat_map = self.original_model(dummy)
            num_features = self.pool(feat_map).view(1, -1).size(1)

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.original_model(x)
        x = self.pool(x)
        x = self.classification_head(x)
        return x

class SimCLRModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super().__init__()
        self.original_model = original_model
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.original_model(x)
        x = self.classification_head(x)
        return x

class LDLModel(nn.Module):
    def __init__(self, original_model, num_classes, model_args):
        super().__init__()
        from LDL import model_select
        untrained_model = model_select(model_args)
        self.backbone = original_model
        num_features = untrained_model.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


# ===== MAIN LOOP =====
for weight_name, model_type in zip(weight_names, model_types):
    print(f"\n=== Processing model: {model_type} ({weight_name}) ===")

    # Directory for images
    model_dir = f"./imgs/{weight_name[:-4]}"
    os.makedirs(model_dir, exist_ok=True)

    # Load model dynamically
    if model_type == "Autoencoder":
        from Autoencoder import Network, model_select
        from Autoencoder_args import conf as model_conf
    elif model_type == "SimCLR":
        from SimCLR import Network, model_select
        from SimCLR_args import conf as model_conf
    elif model_type == "LDL":
        from LDL import Network, model_select
        from LDL_args import conf as model_conf
    else:
        from Autoencoder import model_select
        model = model_select(args)

    if model_type in ["Autoencoder", "SimCLR", "LDL"]:
        model_args = model_conf()
        model = Network(model_args)

    # Wrap model appropriately
    if model_type == "Autoencoder":
        model = AutoencoderModel(model.encoder, numof_classes)
    elif model_type == "SimCLR":
        model = SimCLRModel(model.encoder, numof_classes)
    elif model_type == "LDL":
        model = LDLModel(model.backbone, numof_classes, model_args)

    # Load checkpoint
    checkpoint = torch.load(f"data/weight/{weight_name}", map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    # Extract metrics
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    top1_val_acc = checkpoint.get('top1_val_acc', [])
    top5_val_acc = checkpoint.get('top5_val_acc', [])

    # Save summary stats
    final_val_loss = val_losses[-1] if val_losses else None
    final_top1 = top1_val_acc[-1] if top1_val_acc else None
    final_top5 = top5_val_acc[-1] if top5_val_acc else None
    summary_data.append([model_type, final_val_loss, final_top1, final_top5])

    # Store for combined plots
    all_top1_val_acc[model_type] = top1_val_acc
    all_top5_val_acc[model_type] = top5_val_acc

    # Individual plots
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Loss - {model_type}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{model_dir}/loss.png")
    plt.close()


# ===== Combined Accuracy Plots =====
plt.figure(figsize=(10, 6))
for model_type, acc in all_top1_val_acc.items():
    plt.plot(acc, label=f"{model_type} Top-1")
plt.title("Validation Top-1 Accuracy (All Models)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./imgs/all_models_top1_acc.png")
plt.close()

plt.figure(figsize=(10, 6))
for model_type, acc in all_top5_val_acc.items():
    plt.plot(acc, label=f"{model_type} Top-5")
plt.title("Validation Top-5 Accuracy (All Models)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./imgs/all_models_top5_acc.png")
plt.close()


# ===== Summary Table =====
import pandas as pd
summary_df = pd.DataFrame(summary_data,
                          columns=["Model Type", "Final Val Loss", "Final Top-1 Acc", "Final Top-5 Acc"])
print("\n=== Final Model Summary ===")
print(summary_df)
summary_df.to_csv("./imgs/model_summary.csv", index=False)
