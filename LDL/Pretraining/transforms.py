import torch
from PIL import ImageFilter

class RandomGaussianBlur(object):
    def __init__(self, probability=0.3, min_radius = 2, max_radius = 4):
        self.probability = probability
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, img):
        if torch.rand(1).item() < self.probability:
            radius = torch.randint(self.min_radius, self.max_radius + 1, (1,)).item()
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
