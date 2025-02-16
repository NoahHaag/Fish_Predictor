import os
import re

def split_fish_name(name):
    """
    Attempts to insert a space between the fish species name if it was originally formatted as CamelCase.
    Example: 'SergentMajor_123' → 'Sergent Major 123'
    """
    name = name.replace("_", " ")  # Replace underscores with spaces

    # Add space between words in camel case (e.g., "SergentMajor" → "Sergent Major")
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)

    # Ensure numbers are correctly spaced (e.g., "Sergent Major123" → "Sergent Major 123")
    name = re.sub(r'(\d+)', r' \1', name)

    # Remove extra spaces between words (e.g., "Sergent  Major   123" → "Sergent Major 123")
    name = re.sub(r'\s+', ' ', name)

    return name.strip()

def rename_folders_and_images(train_folder):
    """Renames both subfolders and images by ensuring proper spacing in fish names and numbers."""

    # Rename subfolders (class names)
    for class_name in os.listdir(train_folder):
        class_path = os.path.join(train_folder, class_name)

        if not os.path.isdir(class_path):
            continue  # Skip files

        new_class_name = split_fish_name(class_name)
        new_class_path = os.path.join(train_folder, new_class_name)

        if class_name != new_class_name:
            os.rename(class_path, new_class_path)
            print(f"Renamed folder: '{class_name}' → '{new_class_name}'")

        # Rename images inside the subfolder
        for image_name in os.listdir(new_class_path):
            old_image_path = os.path.join(new_class_path, image_name)

            if not os.path.isfile(old_image_path):
                continue  # Skip directories

            # Extract filename and extension
            name_part, ext = os.path.splitext(image_name)
            ext = ext.lower()

            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                continue  # Skip non-image files

            # Format image name with correct spacing
            new_image_name = split_fish_name(name_part) + ext
            new_image_path = os.path.join(new_class_path, new_image_name)

            if image_name != new_image_name:
                os.rename(old_image_path, new_image_path)
                print(f"Renamed image: '{image_name}' → '{new_image_name}'")

# Example usage:
train_folder = "train/"  # Replace with your actual train folder path
rename_folders_and_images(train_folder)
