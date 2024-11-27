import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from Model import *
import os

# Predefined variables
weight_name = 'checkpoint_epoch_LDL_250_resnet18_rwave-1024.pth.tar'
model_type = 'resnet18'
numof_classes = args.numof_classes  
max_layer = 20

path2model = f'data/weight/{weight_name}'
os.mkdir("./imgs/" + "/" + weight_name[:-4])

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

checkpoint = torch.load(path2model, map_location=device)

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
lr = checkpoint.get('lr', [])

plt.figure(figsize=(16, 16))
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['train', 'val'])
plt.title('Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("imgs/" + "/" + weight_name[:-4] + '/loss.png')
plt.close()
plt.figure(figsize=(16, 16))
plt.plot(lr)
plt.legend(['learning rate'])
plt.title('Learning rate')
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.savefig("imgs/" + "/" + weight_name[:-4] + '/lr.png')
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
            plt.imshow(m[j, 0].detach().cpu().numpy(), cmap='viridis')
            plt.axis('off')
        plt.savefig("imgs/" + "/" + weight_name[:-4] + '/_layer_' + str(i) + '_filters.png')
        plt.close()
        layer += 1

# Run visualization
visualize_filters(model, weight_name, max_layer)