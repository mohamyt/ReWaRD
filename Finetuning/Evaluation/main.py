import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from Model import *
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
import numpy as np
from torchvision.datasets import ImageFolder
from multiprocessing import freeze_support
from torchview import draw_graph


# Predefined variables
weight_name = 'checkpoint_SimCLR_epoch_240_resnet18_ImageNet1k.pth.tar'
model_type = 'resnet18'
if not os.path.isdir("./imgs/" + "/" + weight_name[:-4]):
    os.mkdir("./imgs/" + "/" + weight_name[:-4])
numof_classes = args.numof_classes  
max_layer = 1

path2model = f'data/weight/{weight_name}'

# Configuration and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
def load_model(model_args):
    model_args = conf()
    model = Network(model_args)
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

checkpoint = torch.load(path2model, map_location=device)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CUDA setup
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.deterministic = True
random.seed(args.seed)
torch.manual_seed(args.seed)

if args.path2valdb == args.path2traindb:
    train_portion = 0.8  # Split the training data for validation

val_dataset = ImageFolder(args.path2valdb, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


# Adjust the keys of state_dict if necessary
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  
    new_state_dict[new_key] = v


# Load state_dict into the model
model.load_state_dict(new_state_dict)
model.to(device)
args.start_epoch = checkpoint['epoch'] + 1
train_losses = checkpoint.get('train_losses', [])
batch_losses = checkpoint.get('batch_losses', [])
val_batch_losses = checkpoint.get('val_batch_losses', [])
val_losses = checkpoint.get('val_losses', [])
lr = checkpoint.get('lr', [])
top1_train_acc= checkpoint.get('top1_train_acc', [])
top1_val_acc= checkpoint.get('top1_val_acc', [])
top5_train_acc= checkpoint.get('top5_train_acc', [])
top5_val_acc= checkpoint.get('top5_val_acc', [])

plt.figure(figsize=(16, 16))
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['train', 'val'])
plt.title('Epoch Loss for pretrained ' + args.model_type + ' model with ' + args.usenet + ' architecture finetuned on ImageNet1k dataset' if args.model_type in ['Autoencoder', 'SimCLR', 'LDL'] else 'Epoch Loss for unpretrained model trained on ImageNet1k dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("./imgs/" + "/" + weight_name[:-4] + '/loss.png')
plt.close()
plt.figure(figsize=(16, 16))
plt.plot(lr)
plt.legend(['learning rate'])
plt.title('Learning rate for pretrained ' + args.model_type + ' model with ' + args.usenet + ' architecture finetuned on ImageNet1k dataset' if args.model_type in ['Autoencoder', 'SimCLR', 'LDL'] else 'Learning rate for unpretrained model trained on ImageNet1k dataset')
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.savefig("./imgs/" + "/" + weight_name[:-4] + '/lr.png')
plt.figure(figsize=(16, 16))
plt.plot(top1_train_acc)
plt.plot(top1_val_acc)
plt.plot(top5_train_acc)
plt.plot(top5_val_acc)
plt.legend(['top1_train_acc', 'top1_val_acc', 'top5_train_acc', 'top5_val_acc'])
plt.title('Accuracy for pretrained ' + args.model_type + 'model with ' + args.usenet + ' architecture finetuned on ImageNet1k dataset' if args.model_type in ['Autoencoder', 'SimCLR', 'LDL'] else 'Accuracy for unpretrained model trained on ImageNet1k dataset')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("./imgs/" + "/" + weight_name[:-4] + '/acc.png')
plt.close()

# Load state_dict into the model
model.load_state_dict(new_state_dict)

# Function to visualize filters
def visualize_filters(model, weight_name, max_layer):
    model_weights = []
    conv_layers = []
    model_children = list(model.modules())

    # Counter to keep count of the conv layers
    counter = 0
    # Append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

    print(f"Total convolutional layers: {counter}")

    # Visualize all conv layer filters
    layer = 0
    for i, m in enumerate(model_weights):
        if layer > max_layer:
            break
        plt.figure(figsize=(16, 16))
        num_filters = m.size(0)
        rows = math.ceil(math.sqrt(num_filters))
        cols = math.ceil(num_filters / rows)
        for j in range(num_filters):
            plt.subplot(rows, cols, j + 1)
            weight = m[j].detach().cpu().numpy()  # shape: (C, H, W)
            if weight.shape[0] == 3:
                w = weight - weight.min()
                w = w / (w.max() + 1e-5)
                w = np.transpose(w, (1, 2, 0))  # (H, W, C)
                plt.imshow(w)
            else:
                plt.imshow(weight[0], cmap='gray')

            plt.axis('off')
        plt.suptitle(
            f'Filters of pretrained '
            f'{args.model_type} with {args.usenet} architecture' if args.model_type in ['Autoencoder', 'SimCLR', 'LDL'] else 'Filters of unpretrained model',
            fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("./imgs/" + "/" + weight_name[:-4] + '/_layer_' + str(i) + '_filters.png')
        plt.close()
        layer += 1



# Evaluation
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    val_batch_losses = []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
            imgs, labels = imgs.to(device).float(), labels.to(device).long()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Final metrics
    val_accuracy = correct / total
    print(f"Accuracy: {val_accuracy:.4f}")

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
        # use folder name instead of human-readable label
        class_name = dataset.classes[labels[i].item()]
        plt.title(class_name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def graph_model(model, device="cuda"):
    x = torch.randn(1, 3, 224, 224).to(device)
    model_graph = draw_graph(model, input_data=x, graph_dir="LR", depth=5)
    model_graph.visual_graph.render("./imgs/model_structure", format="pdf")
    print("Saved model graph to ./imgs/model_structure.png")

def brain_score(model):
    from brainscore_vision import benchmark_registry
    print(list(benchmark_registry.keys()))
    #import brainscore_vision

    # benchmark = brainscore_vision.load_benchmark('MajajHong2015public.IT-pls')
    # score = benchmark(model)
    # print(f"Brain-Score: {score.raw:.4f}")



if __name__ == '__main__':
    freeze_support()
    #visualize_batch(val_loader, val_dataset, save_path="./imgs/" + weight_name[:-4] + "/sample_batch.png")
    #evaluate()
    visualize_filters(model, weight_name, max_layer)
    #graph_model(model)
    #brain_score(model)