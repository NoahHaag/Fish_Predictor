import os
import shutil
import time

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets


def move_images_to_species_folders(source_dir, dest_dir):
    """
    Moves images from a large source folder into subfolders based on species name.
    Handles two-part species names with a space separating the name and number.

    Args:
    - source_dir (str): Path to the source directory where the images are currently located.
    - dest_dir (str): Path to the destination directory where the species folders will be created.
    """
    # Loop through all the files in the source directory
    for filename in os.listdir(source_dir):
        # Make sure it's an image file (you can add more file extensions if needed)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract species name from the filename (assuming format "species_name number")
            # Split by space to handle "species_name number" format
            species_name = filename.split(' ')[0]

            # Create the destination folder for the species if it doesn't exist
            species_folder = os.path.join(dest_dir, species_name)
            if not os.path.exists(species_folder):
                os.makedirs(species_folder)

            # Move the image to the correct species folder
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(species_folder, filename)
            shutil.move(source_path, dest_path)
            print(f"Moved {filename} to {species_folder}")


def move_images_to_one_folder(source_dir, dest_dir):
    """
    Moves images from species subdirectories into one big folder, preserving their filenames.

    Args:
    - source_dir (str): Path to the source directory where the species folders are located.
    - dest_dir (str): Path to the destination directory where all images will be moved.
    """
    # Loop through all species subdirectories in the source directory
    for species_folder in os.listdir(source_dir):
        species_folder_path = os.path.join(source_dir, species_folder)

        # Check if it's a directory (a species folder)
        if os.path.isdir(species_folder_path):
            # Loop through all files in the species folder
            for filename in os.listdir(species_folder_path):
                # Make sure it's an image file (you can add more file extensions if needed)
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Construct the source and destination paths
                    source_path = os.path.join(species_folder_path, filename)
                    dest_path = os.path.join(dest_dir, filename)

                    # Check if the file already exists in the destination folder to avoid overwriting
                    if os.path.exists(dest_path):
                        # If you want to handle this situation, you can modify this logic (e.g., by renaming the file)
                        print(f"File {filename} already exists in the destination folder. Skipping.")
                        continue

                    # Move the image to the destination folder
                    shutil.move(source_path, dest_path)
                    print(f"Moved {filename} to {dest_dir}")

def clean_class_name(name):
    # Replace spaces with underscores or adjust as needed
    return name.replace(" ", "_")


source_directory = "train_2/"  # Path to the folder with mixed images
destination_directory = "train_2/"  # Path to the folder where species folders should be created


# move_images_to_species_folders(source_directory, destination_directory)


# Define your CNN model (FishClassifierCNN) class here
class FishClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FishClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Compute correct input size for fc1 dynamically
        self.flattened_size = self._compute_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def _compute_flattened_size(self):
        """Run a dummy forward pass to determine the number of features."""
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)  # Simulate an input batch
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]  # Dynamically compute

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.shape[0], -1)  # Flatten dynamically

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Image transformations: resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Flip image randomly
    transforms.RandomRotation(15),  # Rotate randomly
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Distortion
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset paths
train_dir = "train"  # Path to the training folder
val_dir = "validation"

# Load dataset using ImageFolder (assuming folder structure is 'species_dir/train/species_name')
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# for images, labels in train_loader:
#     print("Train labels batch:", labels)
# for images, labels in val_loader:
#     print("Validation labels batch:", labels)

# Clean class names in both datasets
train_classes_cleaned = {clean_class_name(cls) for cls in train_dataset.classes}
val_classes_cleaned = {clean_class_name(cls) for cls in val_dataset.classes}

# Combine the cleaned classes
all_species = train_classes_cleaned | val_classes_cleaned
num_classes = len(all_species)

print(f"Number of unique classes after cleaning: {num_classes}")

train_dataset.class_to_idx = {species: i for i, species in enumerate(sorted(all_species))}
val_dataset.class_to_idx = train_dataset.class_to_idx

# Save class names to a file
joblib.dump(train_dataset.classes, 'class_names.joblib')

# Set up the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifierCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Decay LR by 0.5 every 5 epochs


# Initialize lists to store loss values for plotting
train_losses = []
val_losses = []

# Define model path
model_path = "CNN models/fish_classifier.pth"

# Check if a previous model exists and load it
if os.path.exists(model_path):
    print("Loading existing model for continued training...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
else:
    print("No previous model found. Training from scratch...")

# Training loop
num_epochs = 20
best_val_loss = float("inf")
patience = 3  # Stop training if no improvement after 3 epochs
patience_counter = 0

for epoch in range(num_epochs):
    start_time = time.time()  # Start timing

    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)

    end_time = time.time()  # End timing
    epoch_duration = end_time - start_time  # Compute duration

    # Convert epoch duration to minutes and seconds
    minutes = int(epoch_duration // 60)
    seconds = int(epoch_duration % 60)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {minutes}m {seconds}s, LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter
        print(f"New best model found! Saving to {model_path}...")
        torch.save(model.state_dict(), model_path)
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    # Step the scheduler
    scheduler.step()
# Plot the loss curves
epochs_ran = len(train_losses)  # Get actual number of epochs completed

plt.plot(range(1, epochs_ran + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs_ran + 1), val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("CNN training loss")
plt.show()
# Save the trained model
torch.save(model.state_dict(), model_path)

print("Model training completed and saved as 'fish_classifier.pth'")
