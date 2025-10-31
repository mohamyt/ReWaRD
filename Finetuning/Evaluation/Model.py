import torch
import torch.nn as nn
import torch.nn.functional as F

from args import conf
from alex import bn_alexnet
from vgg import vgg16_bn, vgg19_bn
from resnet import *
from resnext import *
from densenet  import * 

args = conf()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Selecting the autoencoder
def model_select(args):
    if args.usenet == "bn_alexnet":
        model = bn_alexnet(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "vgg16":
        model = vgg16_bn(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "vgg19":
        model = vgg19_bn(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet18":
        model = resnet18(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet34":
        model = resnet34(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet50":
        model = resnet50(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet101":
        model = resnet101(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet152":
        model = resnet152(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet200":
        model = resnet200(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnext101":
        model = resnext101(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "densenet161":
        model = densenet161(pretrained=False, num_classes=args.numof_classes).to(device)
        return model


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        # Load a pretrained CNN model as backbone, like ResNet
        self.backbone = model_select(args)
        num_labels = args.numof_classes
        num_features = self.backbone.fc.in_features  # Extract features of final layer
        self.backbone.fc = nn.Identity()  # Remove last fully connected layer

        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer to predict label distributions
        self.fc = nn.Linear(num_features, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)  # Feature extraction
        x = self.dropout(x)
        x = self.fc(x)         # Distribution prediction
        return self.softmax(x)  # Apply softmax to get probabilities


