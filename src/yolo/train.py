from src.yolo.model import YOLOv1
from src.yolo.loss import YOLOLoss
from src.yolo.dataloader import VOCDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from src.yolo.transform import *
from src.yolo.earlystopping import EarlyStopping

# Model, loss function, and optimizer
model = YOLOv1(num_classes=3)
criterion = YOLOLoss(S=7, B=2, C=3)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training loop
def train_yolo(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=50, device='cuda'):
    model.train()
    early_stopping = EarlyStopping(patience=15, min_delta=0.00)
    for epoch in range(num_epochs):
        # training on train data 
        running_loss = 0.0
        for images, targets in train_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss = running_loss / len(train_dataloader)

        # Evaluation in eval data
        val_running_loss = 0.0
        for images, targets in val_dataloader:
            images = images.to(device)
            targets = targets.to(device)

            ## Forward Passes
            predictions = model(images)
            loss = criterion(predictions, targets)

            val_running_loss += loss.item()

        val_running_loss = val_running_loss / len(val_dataloader)
        ## Early Stopping Criteria Check
        early_stopping(loss, model, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss}, Validation Loss: {val_running_loss}")

        if early_stopping.early_stopping:
            print("Early Stopping")
            return running_loss, val_running_loss


    print("Training complete!")
    torch.save(model.state_dict(), f"./dump/checkpoint/checkpoint.pt")
    return running_loss, val_running_loss

# Assuming 'train_loader' is your DataLoader
# train_yolo(model, criterion, optimizer, train_loader
train_transform = ComposeTransforms([
    Resize((448, 448)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = ComposeTransforms([
    Resize((448, 448)),
    # RandomHorizontalFlip(),
    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_image_dir = "/home/user/YOLO/data/train_zip/train"
train_annotation_dir = "/home/user/YOLO/data/train_zip/train"

test_image_dir = "/home/user/YOLO/data/test_zip/test"
test_annotation_dir = "/home/user/YOLO/data/test_zip/test"

train_datasets = VOCDataset(image_dir=train_image_dir, annotation_dir=train_annotation_dir, transform=train_transform, C=3, B=2)
train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=16)

test_datasets = VOCDataset(image_dir=test_image_dir, annotation_dir=test_annotation_dir, transform=test_transform, C=3, B=2)
test_dataloader = DataLoader(test_datasets, shuffle=True, batch_size=16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device=device)
train_yolo(model=model, criterion=criterion, optimizer=optimizer, train_dataloader=train_dataloader, val_dataloader=test_dataloader, device=device)

