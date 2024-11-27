import torch
from PIL import ImageFilter

class RandomGaussianBlur(object):
    def __init__(self, probability=0.3, radius=2):
        self.probability = probability
        self.radius = radius

    def __call__(self, img):
        if torch.rand(1).item() < self.probability:
            return img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img
