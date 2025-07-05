import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.models import swin_t
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

# ✅ MMIM model definition (must match training script)
class MMIM(nn.Module):
    def __init__(self, num_classes=9):
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

# ✅ Config
model_path = 'MMIM_best.pth'
test_dir = 'test'
batch_size = 32

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Load test dataset
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MMIM(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Evaluate on test set
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="🔍 Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ Metrics
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"🎯 F1 Score (weighted): {f1:.4f}")
print("\n📝 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ✅ Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as confusion_matrix.png")

# ✅ ROC Curve Plotting
y_true = label_binarize(all_labels, classes=list(range(len(class_names))))
all_probs = np.array(all_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
print("✅ ROC curve saved as roc_curve.png")

# ✅ Predict a single image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Example usage
example_image = os.path.join(test_dir, class_names[0], os.listdir(os.path.join(test_dir, class_names[0]))[0])
print(f"\n🖼️ Example image prediction: {example_image}")
print("👉 Predicted class:", predict_image(example_image))
