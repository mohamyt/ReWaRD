import torch
import torch.nn as nn
import torch.nn.functional as F

from args import conf
from alex import bn_alexnet
from vgg import vgg16_bn, vgg19_bn
from resnet import *
from resnext import *
from densenet import *

args = conf()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_select(args):
    """Return the chosen backbone architecture."""
    if args.usenet == "bn_alexnet":
        return bn_alexnet(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "vgg16":
        return vgg16_bn(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "vgg19":
        return vgg19_bn(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnet18":
        return resnet18(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnet34":
        return resnet34(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnet50":
        return resnet50(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnet101":
        return resnet101(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnet152":
        return resnet152(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnet200":
        return resnet200(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "resnext101":
        return resnext101(pretrained=False, num_classes=args.numof_classes).to(device)
    elif args.usenet == "densenet161":
        return densenet161(pretrained=False, num_classes=args.numof_classes).to(device)
    else:
        raise ValueError(f"Unknown architecture: {args.usenet}")


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        model = model_select(args)

        # Strip off classification layers dynamically
        if hasattr(model, 'fc'):
            # Typical ResNet / ResNeXt
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif hasattr(model, 'classifier'):
            # VGG / AlexNet / DenseNet
            self.encoder = nn.Sequential(*list(model.features.children()))
        else:
            raise ValueError("Unknown model structure")

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, args.r_crop_size, args.r_crop_size).to(device)
            feat = self.forward_features(dummy_input)
            self.feature_dim = feat.shape[1]

    def forward_features(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self.forward_features(x)

    def get_feature_dim(self):
        return self.feature_dim


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = Encoder(args)

        self.fc = nn.Sequential(
            nn.Linear(self.encoder.get_feature_dim(), args.out_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(args.out_dim * 4, args.out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return F.normalize(x, dim=-1)
