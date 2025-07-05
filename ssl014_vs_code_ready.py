import os
import zipfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import swin_t

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Step 1: Set local paths
zip_path = r"C:\Users\student\Desktop\WeedClassifier\datasets.zip"
extract_base = r"C:\Users\student\Desktop\WeedClassifier"
train_dir = r"C:\Users\student\Desktop\WeedClassifier\weed_dataset_split\train"
test_dir = r"C:\Users\student\Desktop\WeedClassifier\weed_dataset_split\test"
checkpoint_path = r"C:\Users\student\Desktop\WeedClassifier\MMIM_checkpoints"
os.makedirs(checkpoint_path, exist_ok=True)

# Step 2: Extract zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_base)

# Step 3: Find dataset directory
def find_dataset_dir(base_path):
    for root, dirs, files in os.walk(base_path):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            return os.path.dirname(root)
    subdirs = os.listdir(base_path)
    if len(subdirs) == 1:
        return os.path.join(base_path, subdirs[0])
    return base_path

dataset_dir = find_dataset_dir(extract_base)
print(f"✅ Found dataset directory at: {dataset_dir}")

# Step 4: Split and copy
def split_and_save_dataset(source_dir, train_dir, test_dir, test_size=0.2):
    class_names = os.listdir(source_dir)
    total_train = total_test = 0

    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(images) == 0:
            print(f"⚠️ Skipping empty class: {class_name}")
            continue

        print(f"📂 Processing class '{class_name}' with {len(images)} images...")
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

        total_train += len(train_imgs)
        total_test += len(test_imgs)
        print(f"✅ Copied {len(train_imgs)} train, {len(test_imgs)} test images for '{class_name}'")

    print(f"\n🎯 TOTAL: {total_train} train images, {total_test} test images")
    print(f"✅ Split complete! Data saved in: {train_dir} and {test_dir}")

split_and_save_dataset(dataset_dir, train_dir, test_dir)

# Step 5: Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Step 6: Dataset & DataLoaders
dataset = ImageFolder(train_dir, transform=transform)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Step 7: Model
class MMIM(nn.Module):
    def __init__(self, num_classes=36):
        super(MMIM, self).__init__()
        self.backbone = swin_t(weights='IMAGENET1K_V1')
        self.backbone.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MMIM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss()

# Step 8: Training and validation loops
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"🟢 Training Epoch {epoch}", leave=False)
    for imgs, labels in progress_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"🔵 Validating Epoch {epoch}", leave=False)
    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader.dataset)

# Step 9: Training loop with early stopping
def main():
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs = 50
    patience = 5

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        print(f"📊 Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}\\MMIM_epoch{epoch}.pth")
            print(f"💾 Model checkpoint saved at epoch {epoch}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{checkpoint_path}\\MMIM_best.pth")
            print("🏅 New best model saved")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered.")
            break

if __name__ == '__main__':
    main()
