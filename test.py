import os


def remove_underscores(name):
    """Replace underscores with spaces in class names and image names."""
    return name.replace("_", " ")


def rename_classes_and_images(train_folder):
    """Renames class folders and image files by replacing underscores with spaces."""

    for class_name in os.listdir(train_folder):
        class_path = os.path.join(train_folder, class_name)

        if not os.path.isdir(class_path):
            continue  # Skip files

        # Rename the folder
        new_class_name = remove_underscores(class_name)
        new_class_path = os.path.join(train_folder, new_class_name)

        if class_name != new_class_name:
            os.rename(class_path, new_class_path)
            print(f"Renamed folder: '{class_name}' → '{new_class_name}'")
        else:
            print(f"Folder already correct: '{class_name}'")

        # Rename images inside the folder
        for image_name in os.listdir(new_class_path):
            old_image_path = os.path.join(new_class_path, image_name)

            if not os.path.isfile(old_image_path):
                continue  # Skip directories

            # Extract file extension
            name_part, ext = os.path.splitext(image_name)
            ext = ext.lower()

            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                continue  # Skip non-image files

            # Rename image files
            new_image_name = remove_underscores(name_part) + ext
            new_image_path = os.path.join(new_class_path, new_image_name)

            if image_name != new_image_name:
                os.rename(old_image_path, new_image_path)
                print(f"Renamed image: '{image_name}' → '{new_image_name}'")


# Example usage:
train_folder = "train_2/"  # Replace with your actual train folder path
rename_classes_and_images(train_folder)
