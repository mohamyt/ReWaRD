import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from Model import *
import os

# Predefined variables
weight_name = 'checkpoint_SimCLR_epoch_60_resnet18_rwave-1024.pth.tar'
model_type = 'resnet18'
numof_classes = 10  
max_layer = 20

os.mkdir("./imgs/" + "/" + weight_name[:-4])

# Configuration and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the network and weights
model = Network(args).to(device)
checkpoint = torch.load(f"data/weight/resnet18/{weight_name}", map_location=device)

state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
args.start_epoch = checkpoint['epoch'] + 1
train_losses = checkpoint.get('train_losses', [])
batch_losses = checkpoint.get('batch_losses', [])
val_batch_losses = checkpoint.get('val_batch_losses', [])
val_losses = checkpoint.get('val_losses', [])

plt.figure(figsize=(16, 16))
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['train', 'val'])
plt.title('Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("imgs/" + "/" + weight_name[:-4] + '/loss.png')
plt.close()

# Load state_dict into the model
model.load_state_dict(new_state_dict)
model = model.encoder

# Function to visualize filters
def visualize_filters(model, weight_name, max_layer):
    model_weights = []
    conv_layers = []
    model_children = list(model.modules())

    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

    print(f"Total convolutional layers: {counter}")

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

visualize_filters(model, weight_name, max_layer)