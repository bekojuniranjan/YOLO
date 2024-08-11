import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for t in self.transforms:
            image, bboxes = t(image, bboxes)
        return image, bboxes

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bboxes):
        width, height = image.size
        image = F.resize(image, self.size)
        scale_x = self.size[0] / width
        scale_y = self.size[1] / height
        bboxes = [[b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y] for b in bboxes]
        return image, bboxes

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            width = image.size[0]
            bboxes = [[width - b[2], b[1], width - b[0], b[3]] for b in bboxes]
        return image, bboxes

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, bboxes):
        image = self.transform(image)
        return image, bboxes

class ToTensor:
    def __call__(self, image, bboxes):
        image = F.to_tensor(image)
        bboxes = torch.tensor(bboxes)
        return image, bboxes

class Normalize:
    def __init__(self, mean, std):
        self.transform = T.Normalize(mean, std)

    def __call__(self, image, bboxes):
        image = self.transform(image)
        return image, bboxes


if __name__ == "__main__":
    # Example of combining the transforms
    transform = ComposeTransforms([
        Resize((448, 448)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Usage in your Dataset class
    # transformed_image, transformed_bboxes = transform(image, bboxes)
