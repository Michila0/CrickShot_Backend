import torch
import torchvision #Core deep learning libraries
from torchvision import transforms #Image transformation
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler # Data handling
from PIL import Image #Image processing
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt #Visualization
import numpy as np #Numerical operation
from sklearn.metrics import confusion_matrix, classification_report # Evaluation metrics
import seaborn as sns
import torch.optim.lr_scheduler
from collections import Counter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

"""Custom Dataset Class"""
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths #collected data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


"""Data Preparation and Visualization"""

data_dir = "./data"
image_paths = []
labels = []
class_names = ['drive', 'legglance-flick', 'pullshot', 'sweep']
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# Load image paths and labels
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_paths.append(os.path.join(class_dir, img_file)) # Store image path
            labels.append(class_to_idx[class_name]) # Store numeric label

# Check class distribution
print("Class distribution:", Counter(labels))

# Convert to numpy arrays for K-Fold
all_image_paths = np.array(image_paths)
all_labels = np.array(labels)


# Visualize sample images
def show_images(dataset, num_images=6):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 10))
    for i in range(num_images):
        image, label = dataset[i]
        ax = axes[i]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(class_names[label])
        ax.axis('off')
    plt.show()


"""## Data Transformations"""

# Stronger data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((150, 150)), # Resize images
    transforms.RandomHorizontalFlip(), # Data augmentation
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), #Random shifts
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), #Random color adjustments
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(), # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

# Simple transform for validation and test
val_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""Model Definition with Regularization"""
def create_model():
    model = torchvision.models.resnet50(pretrained=True)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False  # Freeze layers

    # Replace final layer with dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential( #Replace final layer
        nn.Dropout(0.5),    #regularization
        nn.Linear(num_ftrs, 4))

    return model


"""Training and Evaluation Functions"""
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, patience=5):
    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training Process
    for epoch in range(num_epochs):
        model.train() # Set training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Reset gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Back-propagate
            optimizer.step() # Update weight

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1) #shot Classification
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 50)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model weights
    model.load_state_dict(best_model_weights)

    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, criterion):
    model.eval() #set evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Calculate validation metrics
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / len(data_loader)
    accuracy = correct / total

    return loss, accuracy


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


"""K-Fold Cross Validation"""

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = [] # Tracks accuracy per fold

for fold, (train_idx, val_idx) in enumerate(kf.split(all_image_paths)):
    print(f"\n{'=' * 40}")
    print(f"Fold {fold + 1}/5")
    print(f"{'=' * 40}")

    # Create datasets
    train_paths, val_paths = all_image_paths[train_idx], all_image_paths[val_idx]
    train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

    # Create weighted sampler to handle class imbalance
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CustomImageDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, optimizer
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler
    )

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Evaluate on validation set
    val_loss, val_acc = evaluate_model(model, val_loader, criterion)
    print(f"\nFold {fold + 1} Validation Accuracy: {val_acc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(model, val_loader)

    # Save results
    fold_results.append(val_acc)

    # Save the best model
    if fold == 0 or val_acc > max(fold_results[:-1]):
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved new best model")

# Print cross-validation results
print("\nCross-Validation Results:")
print(f"Mean Validation Accuracy: {np.mean(fold_results):.4f}")
print(f"Std Dev: {np.std(fold_results):.4f}")

"""## Final Test Evaluation"""

# Load the best model
best_model = create_model().to(device)
best_model.load_state_dict(torch.load('best_model.pth'))

# Create test set (20% holdout)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

test_dataset = CustomImageDataset(test_paths, test_labels, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate on test set
test_loss, test_acc = evaluate_model(best_model, test_loader, criterion)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Plot confusion matrix
plot_confusion_matrix(best_model, test_loader)

"""## Prediction Visualization"""


def visualize_predictions(model, dataset, num_images=6):
    model.eval()
    indices = np.random.choice(len(dataset), num_images, replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        image, true_label = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, pred_label = torch.max(output, 1)
            pred_label = pred_label.item()

        # Convert image back for display
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Visualize some test predictions
visualize_predictions(best_model, test_dataset)