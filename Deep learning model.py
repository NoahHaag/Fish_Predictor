if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # Ensures correct multiprocessing start method

    import os
    import shutil
    import sys
    import time

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
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


    source_directory = "train/"  # Path to the folder with mixed images
    destination_directory = "train/"  # Path to the folder where species folders should be created


    # move_images_to_species_folders(source_directory, destination_directory)

    def evaluate_model(model, data_loader, device):
        model.eval()  # Set model to evaluation mode
        all_labels = []
        all_preds = []
        with torch.no_grad():  # No gradient tracking during validation
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # Get the predicted class
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1


    def plot_confusion_matrix(y_true, y_pred, class_names, filename="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save the confusion matrix plot
        plt.savefig(filename)
        plt.close()


    # Save both model and best validation accuracy
    def save_model(model, best_val_accuracy, model_path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_val_accuracy': best_val_accuracy
        }, model_path)


    # Load model weights and best validation accuracy
    def load_model(model, model_path):
        checkpoint = torch.load(model_path, weights_only=True)

        # Load the model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Retrieve the best validation accuracy
        best_val_accuracy = checkpoint['best_val_accuracy']

        return model, best_val_accuracy


    # Inception Module
    class InceptionBlock(nn.Module):
        def __init__(self, in_channels):
            super(InceptionBlock, self).__init__()

            # 1x1 Convolution
            self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)

            # 1x1 -> 3x3 Convolution
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1)
            )

            # 1x1 -> 5x5 Convolution
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, padding=2)
            )

            # 3x3 Max Pool -> 1x1 Convolution
            self.branch4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, 32, kernel_size=1)
            )

            self.batch_norm = nn.BatchNorm2d(288)  # Adjust output channels after concatenation
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
            x = self.batch_norm(x)
            return self.dropout(F.relu(x))


    class FishClassifierCNN(nn.Module):
        def __init__(self, num_classes):
            super(FishClassifierCNN, self).__init__()

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.batch_norm1 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)

            # Deeper Inception blocks
            self.inception1 = InceptionBlock(64)  # First Inception Block
            self.inception2 = InceptionBlock(288)  # Second Inception Block (concatenation increases depth)
            self.inception3 = InceptionBlock(288)  # Third Inception Block for more depth

            self.dropout = nn.Dropout(0.5)

            # Compute correct input size for fc1 dynamically
            self.flattened_size = self._compute_flattened_size()

            self.fc1 = nn.Linear(self.flattened_size, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_classes)

        def _compute_flattened_size(self):
            """Run a dummy forward pass to determine the number of features."""
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)  # Simulate an input batch
                x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
                x = self.pool(self.inception1(x))
                x = self.pool(self.inception2(x))
                x = self.pool(self.inception3(x))
                return x.view(1, -1).shape[1]  # Dynamically compute

        def forward(self, x):
            x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
            x = self.pool(self.inception1(x))
            x = self.pool(self.inception2(x))
            x = self.pool(self.inception3(x))

            x = x.view(x.shape[0], -1)  # Flatten dynamically
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)

            return x


    # Image transformations: resizing and normalization
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.RandomRotation(10),  # Small rotations to avoid unrealistic angles
        transforms.RandomAffine(degrees=0, shear=5, scale=(0.9, 1.1)),  # Small affine changes
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Minor adjustments
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),  # Only applied sometimes
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),  # Blur sometimes
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize only
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define dataset paths
    train_dir = "train"  # Path to the training folder
    val_dir = "validation"
    test_dir = "test"

    # Load dataset using ImageFolder (assuming folder structure is 'species_dir/train/species_name')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Visualize transformations ####
    # Load a single image from the dataset
    img_path, _ = train_dataset.samples[3500]  # Get image path
    original_img = Image.open(img_path).convert("RGB")  # Load image

    # Number of transformed samples to visualize
    num_samples = 9

    # Apply transformations multiple times
    transformed_images = [transform(original_img) for _ in range(num_samples)]


    # Convert tensors to numpy images for plotting
    def tensor_to_image(tensor):
        tensor = tensor.cpu().detach()  # Remove gradient tracking
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1,
                                                                                                                       1)  # Unnormalize
        tensor = tensor.numpy().transpose(1, 2, 0)  # Convert to HWC format
        return np.clip(tensor, 0, 1)  # Clip values to [0,1] for display


    # Plot transformed images in a grid
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))  # 3x3 grid
    for ax, img in zip(axes.flatten(), transformed_images):
        ax.imshow(tensor_to_image(img))
        ax.axis("off")

    plt.suptitle("Augmented Images from Transformations", fontsize=14)
    plt.savefig("Augmented images")
    plt.show()

    start_dataloader = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if i == 5:  # Measure only first 5 batches
            break
    end_dataloader = time.time()

    print(f"Average dataloader time per batch: {(end_dataloader - start_dataloader) / 5:.4f} seconds")

    # Clean class names in both datasets
    train_classes_cleaned = {clean_class_name(cls) for cls in train_dataset.classes}
    val_classes_cleaned = {clean_class_name(cls) for cls in val_dataset.classes}
    test_classes_cleaned = {clean_class_name(cls) for cls in test_dataset.classes}

    # Combine the cleaned classes
    all_species = train_classes_cleaned | val_classes_cleaned | test_classes_cleaned
    num_classes = len(all_species)

    print(f"Number of unique classes after cleaning: {num_classes}")

    train_dataset.class_to_idx = {species: i for i, species in enumerate(sorted(all_species))}
    val_dataset.class_to_idx = train_dataset.class_to_idx

    # Save class names to a file
    joblib.dump(train_dataset.classes, 'class_names.joblib')

    # Set up the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FishClassifierCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Initialize lists to store loss values for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Define model path
    model_path = "CNN models/fish_classifier.pth"

    # Check if a previous model exists and load it
    if os.path.exists(model_path):
        print("Loading existing model for continued training...")
        model, best_val_accuracy = load_model(model, model_path)
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    else:
        print("No previous model found. Training from scratch...")
        best_val_accuracy = 0.0


    # Training loop
    num_epochs = 25
    patience_counter = 0
    patience = 3  # Set your patience level
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_train_loss = 0.0
        batch_times = []  # Store batch times

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()  # Track batch start time

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            batch_end = time.time()
            batch_times.append(batch_end - batch_start)  # Store batch processing time

            # Estimate time remaining
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = len(train_loader) - (batch_idx + 1)
            estimated_time_remaining = avg_batch_time * remaining_batches
            est_mins = int(estimated_time_remaining // 60)
            est_secs = int(estimated_time_remaining % 60)

            # Print real-time batch progress
            sys.stdout.write(f"\rEpoch {epoch+1}/{num_epochs} - "
                             f"Batch {batch_idx+1}/{len(train_loader)} - "
                             f"Time per batch: {avg_batch_time:.2f}s - "
                             f"Estimated time left: {est_mins}m {est_secs}s")
            sys.stdout.flush()  # Force immediate output

        print()  # Ensure newline after epoch progress

        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Compute validation metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # Compute epoch duration
        epoch_duration = time.time() - start_time
        avg_epoch_time = epoch_duration / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        total_time_remaining = avg_epoch_time * remaining_epochs
        minutes = int(epoch_duration // 60)
        seconds = int(epoch_duration % 60)

        current_lr = optimizer.param_groups[0]['lr']  # Get learning rate from optimizer
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {minutes}m {seconds}s | "
              f"Estimated total remaining time: {total_time_remaining:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best model found! Saving to {model_path}...")
            save_model(model, val_accuracy, model_path)

        # Check if accuracy improved but loss did not
        elif val_accuracy > (val_accuracies[-1] if val_accuracies else 0):
            print("Warning: Validation accuracy increased but loss did not improve! Possible overfitting detected.")
            patience_counter += 1  # Increase patience counter to monitor behavior

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

        # Store validation accuracy history
        val_accuracies.append(val_accuracy)

        # Step scheduler if defined
        if scheduler:
            scheduler.step(val_loss)

    # Plot the loss curves
    epochs_ran = len(train_losses)  # Get actual number of epochs completed

    # Ensure we actually recorded validation losses before plotting
    if len(val_losses) == 0:
        print("Warning: No validation losses recorded!")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("CNN_training_loss.png")
    plt.show()

    # Save the trained model
    save_model(model, best_val_accuracy, model_path)
    print("Model training completed and saved as 'fish_classifier.pth'")


    test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")


