import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from Autoencoder import *  
import os

# Predefined variables
weight_name = 'checkpoint_epoch_50.pth.tar'
model_type = 'resnet18'
numof_classes = 10  # Example value, replace with your actual number of classes
max_layer = 5  # Example value, adjust as needed

# Configuration and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the network and weights
model = Network(args).to(device)
checkpoint = torch.load('data/weight/resnet18/checkpoint_epoch_50.pth.tar', map_location=device) #adjust path to the weight

state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model = model.encoder

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
        plt.savefig("imgs/" + weight_name[:-4] + '_layer_' + str(i) + '_filters.png')
        plt.close()
        layer += 1

# Run visualization
visualize_filters(model, weight_name, max_layer)