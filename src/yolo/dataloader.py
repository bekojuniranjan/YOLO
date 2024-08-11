import os
import math
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.yolo.transform import *
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, S=7, B=2, C=20, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.image_filenames = [img for img in  os.listdir(image_dir) if img.endswith('.jpg')]
        self.class_idx = {
            'apple': 0,
            'orange': 1,
            'banana': 2,
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        annotation_path = os.path.join(self.annotation_dir, image_filename.replace(".jpg", ".xml"))

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load and parse annotation
        boxes, labels = self.parse_annotation(annotation_path)

        # transforms
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert to YOLO format: grid, bounding boxes, and class labels
        target = self.encode_to_yolo_format(boxes, labels, image.shape[1], image.shape[2])

        return image, target

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels

    def encode_to_yolo_format(self, boxes, labels, image_height, image_width):
        # Initialize empty target tensor
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        # Convert bounding boxes and labels to grid format
        for box, label in zip(boxes, labels):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]

            # Normalize coordinates by image size (assume image size is known)
            x_center /= image_width
            y_center /= image_height
            width = width.type(torch.float)
            height = height.type(torch.float)
            width /= image_width
            height /= image_height

            # Determine grid cell
            grid_x = math.floor(x_center * self.S)
            grid_y = math.floor(y_center * self.S)
            
            # Relative to grid cell
            x_center = x_center * self.S - grid_x
            y_center = y_center * self.S - grid_y

            # Fill in the first bounding box slot (index 0 to 4)
            if target[grid_y, grid_x, 4] == 0:  # Check if this grid cell already has an object
                target[grid_y, grid_x, :5] = torch.tensor([x_center, y_center, width, height, 1])
            
            # Optionally fill in the second bounding box slot (index 5 to 9)
            elif target[grid_y, grid_x, 9] == 0:
                target[grid_y, grid_x, 5:10] = torch.tensor([x_center, y_center, width, height, 1])
            
            target[grid_y, grid_x, 10 + self.class_idx[label]] = 1  # Set the class label
        
        return target



if __name__ == "__main__":
    transform = ComposeTransforms([
        Resize((448, 448)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # transform = None
    image_dir = "/home/fm-pc-lt-278/deep_train/computer vision/deep_learning/CNN/object detection/YOLO/data/train_zip/train"
    annotation_dir = "/home/fm-pc-lt-278/deep_train/computer vision/deep_learning/CNN/object detection/YOLO/data/train_zip/train"

    datasets = VOCDataset(image_dir=image_dir, annotation_dir=annotation_dir, transform=transform, C=3)
    print(datasets[0])
    