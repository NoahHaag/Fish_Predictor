import os

import cv2


def augment_and_save_images(input_folder, output_folder, num_augmentations=10):
    """
    Ensures each original image in the dataset has exactly num_augmentations augmented images.
    Skips creating augmentations for images that already meet the requirement or are invalid (NoneType).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the augmentation pipeline
    import imgaug.augmenters as iaa

    aug_pipeline = iaa.Sequential([
        # Geometric Augmentations (Shape & Position)
        iaa.Affine(rotate=(-25, 25)),  # Increase rotation range slightly
        iaa.Fliplr(0.5),  # Horizontal flip (common for fish)
        iaa.Flipud(0.2),  # Occasional vertical flip
        iaa.ScaleX((0.85, 1.15)),  # Scale width
        iaa.ScaleY((0.85, 1.15)),  # Scale height
        iaa.PiecewiseAffine(scale=(0.01, 0.03)),  # Slight warping to simulate underwater distortions

        # Color Augmentations (Lighting & Hue Variations)
        iaa.Multiply((0.80, 1.20)),  # Adjust brightness more
        iaa.AddToHueAndSaturation((-10, 10)),  # Larger hue/saturation shifts
        iaa.Grayscale(alpha=(0.0, 0.2)),  # Some desaturation (helps generalization)

        # Texture & Noise Augmentations (Underwater Distortions)
        iaa.GaussianBlur(sigma=(0.0, 1.5)),  # Slight blur variation
        iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255)),  # Slightly increased noise

        # Edge & Feature Augmentations (Stripes, Spots, etc.)
        iaa.Sharpen(alpha=(0.0, 0.4)),  # More edge sharpening
        iaa.Emboss(alpha=(0.0, 0.3)),  # Subtle embossing
        iaa.ContrastNormalization((0.75, 1.25)),  # Adjust contrast
    ])

    # Preload existing augmented files
    existing_files = set(os.listdir(output_folder))

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if not filename.endswith(".jpeg"):
            continue

        # Extract species name and current ID from the original filename
        species, current_id_str = " ".join(filename.rsplit(" ", 1)[:-1]), filename.rsplit(" ", 1)[-1].split(".")[0]
        base_filename = f"{species} {current_id_str}"

        # Count existing augmentations for this original image
        existing_augmented = [
            f for f in existing_files if f.startswith(base_filename)
        ]
        num_existing_augmented = len(existing_augmented)

        # Skip if there are already enough augmentations
        if num_existing_augmented >= num_augmentations:
            print(f"Skipping {filename}, already has {num_existing_augmented} augmentations.")
            continue

        # Load the original image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Check for NoneType (invalid image)
        if image is None:
            print(f"Skipping invalid image: {filename}")
            continue

        # Generate only the missing augmentations
        for augmentation_id in range(num_existing_augmented + 1, num_augmentations + 1):
            # Apply augmentation
            augmented_image = aug_pipeline(image=image)

            # Create a unique filename for the augmented image
            new_filename = f"{species} {current_id_str} aug{augmentation_id}.jpeg"
            output_path = os.path.join(output_folder, new_filename)

            # Save the augmented image
            cv2.imwrite(output_path, augmented_image)
            existing_files.add(new_filename)  # Update existing file list
            print(f"Saved augmented image: {output_path}")

    print(f"All augmentations saved to {output_folder}")

def delete_empty_files(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")


# Input and output folder paths
input_folder = "train"
input_folder2 = "train_2"
output_folder = "train_augmented"

# Delete empty files before starting the image generation
delete_empty_files(output_folder)

# Perform augmentation
augment_and_save_images(input_folder, output_folder)
augment_and_save_images(input_folder2, output_folder)
