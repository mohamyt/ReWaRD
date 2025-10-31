import torch
import torch.nn as nn

from args import conf
from alex import bn_alexnet
from vgg import vgg16_bn, vgg19_bn
from resnet import *
from resnext import *
from densenet  import * 

args = conf()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        model = model_select(args)
        self.encoder = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, encoded_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoded_channels, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=1),  
        )

    def forward(self, x, original_size):
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=original_size, mode='bilinear', align_corners=False)  # Resize
        return x


class Network(nn.Module):
    def __init__(self, args, input_size=(args.crop_size, args.crop_size)):
        super(Network, self).__init__()
        self.encoder = Encoder(args)

        # Dummy input to determine encoded size
        dummy_input = torch.randn(1, 3, *input_size).to(device)
        with torch.no_grad():
            encoded_output = self.encoder(dummy_input)
        _, encoded_channels, encoded_height, encoded_width = encoded_output.size()

        self.decoder = Decoder(encoded_channels)

    def forward(self, x):
        original_size = x.size()[2:]  # Save original height and width
        x = self.encoder(x)
        x = self.decoder(x, original_size)  # Pass original size to decoder
        return x
