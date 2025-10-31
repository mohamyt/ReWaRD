# ==========================
# Model.py (FINAL PATCHED)
# ==========================

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


# ------------------------------------------------------------
# Backbone selector
# ------------------------------------------------------------
def model_select(args):
    if args.usenet == "bn_alexnet":
        return bn_alexnet(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "vgg16":
        return vgg16_bn(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "vgg19":
        return vgg19_bn(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnet18":
        return resnet18(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnet34":
        return resnet34(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnet50":
        return resnet50(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnet101":
        return resnet101(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnet152":
        return resnet152(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnet200":
        return resnet200(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "resnext101":
        return resnext101(pretrained=False, num_classes=args.numof_classes)
    elif args.usenet == "densenet161":
        return densenet161(pretrained=False, num_classes=args.numof_classes)
    else:
        raise ValueError(f"Unknown usenet: {args.usenet}")


# ------------------------------------------------------------
# Main network wrapper
# ------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, args, num_labels=args.numof_classes, num_features=2048):
        """
        num_labels: number of output classes (number of clusters).
        If None, uses args.numof_classes.
        """
        super(Network, self).__init__()
        self.backbone = model_select(args)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------
        # Detect feature dimension robustly
        # --------------------------------------

        if hasattr(self.backbone, "fc"):
            # ResNet and similar
            if num_features is None:
                num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif hasattr(self.backbone, "classifier"):
            clf = self.backbone.classifier
            if isinstance(clf, nn.Sequential):
                last = next((m for m in reversed(list(clf)) if isinstance(m, nn.Linear)), None)
                if last is None:
                    raise RuntimeError("No Linear layer found in classifier sequence")
                if num_features is None:
                    num_features = last.in_features
                self.backbone.classifier = nn.Identity()
            elif isinstance(clf, nn.Linear):
                num_features = clf.in_features
                self.backbone.classifier = nn.Identity()

        # Fallback for anything weird/custom
        if num_features is None:
            with torch.no_grad():
                self.backbone.eval()
                dummy = torch.randn(1, 3, args.crop_size, args.crop_size)
                feat = self.backbone(dummy)
                if feat.dim() > 2:
                    feat = self.pool(feat)
                    feat = torch.flatten(feat, 1)
                num_features = feat.size(1)

        self.num_features = num_features
        self.num_labels = int(num_labels) if num_labels is not None else int(args.numof_classes)

        # --------------------------------------
        # Final classifier head
        # --------------------------------------
        self.fc = nn.Linear(self.num_features, self.num_labels)

        print(f"[Model] Using backbone: {args.usenet}")
        print(f"[Model] Detected feature dim: {self.num_features}")
        print(f"[Model] Output classes (clusters): {self.num_labels}")

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):
        x = self.backbone(x)
        if x.dim() > 2:
            x = self.pool(x)
            x = torch.flatten(x, 1)
        # Safety: if the incoming feature dimension doesn't match the fc layer,
        # recreate the fc to avoid matmul shape mismatch (can happen when
        # switching backbone variants or loading a checkpoint with different arch).
        if x.size(1) != self.fc.in_features:
            out_features = self.fc.out_features
            device = next(self.parameters()).device
            print(f"[Model Warning] feature dim changed from {self.fc.in_features} to {x.size(1)}; rebuilding fc")
            self.fc = nn.Linear(x.size(1), out_features).to(device)
        x = self.fc(x)
        # Return raw logits. Loss functions (CrossEntropyLoss or custom soft-label loss)
        # will perform the appropriate log-softmax internally.
        return x
