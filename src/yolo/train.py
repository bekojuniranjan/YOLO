from src.yolo.model import YOLOv1
from src.yolo.loss import YOLOLoss
from src.yolo.dataloader import VOCDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from src.yolo.transform import *
# Model, loss function, and optimizer
model = YOLOv1(num_classes=3)
criterion = YOLOLoss(S=7, B=2, C=3)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_yolo(model, criterion, optimizer, dataloader, num_epochs=50, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    print("Training complete!")

# Assuming 'train_loader' is your DataLoader
# train_yolo(model, criterion, optimizer, train_loader
transform = ComposeTransforms([
    Resize((448, 448)),
    RandomHorizontalFlip(),
    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_dir = "/home/user/YOLO/data/test_zip/test"
annotation_dir = "/home/user/YOLO/data/test_zip/test"

datasets = VOCDataset(image_dir=image_dir, annotation_dir=annotation_dir, transform=transform, C=3, B=2)
dataloader = DataLoader(datasets, shuffle=True, batch_size=16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device=device)
train_yolo(model=model, criterion=criterion, optimizer=optimizer, dataloader=dataloader, device=device)

