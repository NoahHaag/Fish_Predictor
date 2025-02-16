import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import joblib
import mahotas
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.transform import radon
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def count_images_in_folder(folder_path):
    image_count = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Check if the file is an image by attempting to open it
            with Image.open(file_path) as img:
                image_count += 1
        except (IOError, FileNotFoundError):
            # Not an image file, skip it
            continue
    return image_count


def extract_raw_pixels(img_resized):
    """Extract raw pixel data as a flattened array."""
    return img_resized.flatten()


def extract_color_histogram(img_resized):
    """Extract color histogram from the image."""
    hist = cv2.calcHist([img_resized], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    return hist.flatten()


def extract_edge_features(img_resized):
    """Extract edge features using Canny edge detection."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)
    return edges.flatten()


def extract_lbp_features(img_resized):
    """Extract local binary pattern histogram."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return lbp_hist


def extract_haralick_features(img_resized):
    """Extract Haralick texture features."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    haralick = mahotas.features.haralick(gray_img).mean(axis=0)
    return haralick


def extract_gabor_features(img_resized):
    """Extract Gabor filter features."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    gabor_features = []
    for theta in range(4):
        theta = theta * np.pi / 4
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
        gabor_features.append(np.mean(filtered))
        gabor_features.append(np.std(filtered))
    return np.array(gabor_features)


def extract_hog_features(img_resized):
    """Extract HOG features."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    hog_features = cv2.HOGDescriptor().compute(gray_img)
    return hog_features.flatten()


def extract_color_moments(img_resized):
    """Extract mean, standard deviation, and skewness for each color channel."""
    moments = []
    for channel in cv2.split(img_resized):
        mean = np.mean(channel)
        std = np.std(channel)

        # Check if the standard deviation is zero to prevent division by zero in skewness calculation
        if std == 0:
            skew = 0
        else:
            skew = np.mean((channel - mean) ** 3) ** (1 / 3)

        # Check for NaNs or Infs before appending
        if np.isnan(mean) or np.isnan(std) or np.isnan(skew) or np.isinf(mean) or np.isinf(std) or np.isinf(skew):
            mean, std, skew = 0, 0, 0

        moments.append(mean)
        moments.append(std)
        moments.append(skew)
    return np.array(moments)


def extract_dominant_colors(img_resized, n_colors=3):
    """Extract dominant colors using k-means clustering."""
    pixels = img_resized.reshape(-1, 3)

    # Avoid using too many clusters if the dataset has few distinct points
    unique_pixels = np.unique(pixels, axis=0)
    if unique_pixels.shape[0] < n_colors:
        print(
            f"Warning: Only {unique_pixels.shape[0]} unique colors found. Reducing n_clusters to {unique_pixels.shape[0]}")
        n_colors = unique_pixels.shape[0]

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(unique_pixels)
    return kmeans.cluster_centers_.flatten()


def extract_stripe_spot_features(img_resized):
    """Detect stripes or spots using edge detection and clustering."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = sobel(gray_img)  # Edge detection
    edge_density = np.sum(edges) / edges.size
    return np.array([edge_density])


def extract_gradient_features(img_resized):
    """Extract gradient magnitude and orientation statistics."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_orientation = np.arctan2(grad_y, grad_x)

    magnitude_mean = np.mean(grad_magnitude)
    magnitude_std = np.std(grad_magnitude)
    orientation_mean = np.mean(grad_orientation)
    orientation_std = np.std(grad_orientation)

    return np.array([magnitude_mean, magnitude_std, orientation_mean, orientation_std])


def extract_fourier_descriptors(image, image_size=(128, 128)):
    gray_image = cv2.cvtColor(cv2.resize(image, image_size), cv2.COLOR_BGR2GRAY)
    radon_transform = radon(gray_image)
    fourier_desc = np.fft.fft2(radon_transform)
    fourier_magnitude = np.abs(fourier_desc).flatten()
    return fourier_magnitude


def extract_zernike_moments(img_resized, radius=21):
    """Extract Zernike moments for shape and texture analysis."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    zernike = mahotas.features.zernike_moments(gray_img, radius)
    return np.array(zernike)


def extract_aspect_ratio(img_resized):
    """Extract aspect ratio of the image's bounding box."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]  # Assuming the largest contour
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    return np.array([aspect_ratio])


def extract_shape_features(img_resized):
    """Extract shape features like solidity and circularity."""
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]  # Assuming the largest contour
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return np.array([solidity, circularity])


# Precompute feature names and sizes
def get_feature_info(image_size, use_raw_pixels, use_color_histogram, use_edge_features,
                     use_lbp, use_haralick, use_gabor, use_hog,
                     use_color_moments, use_dominant_colors, use_stripe_spot, use_gradient, use_fourier_descriptors,
                     use_zernike_moments, use_aspect_ratio, use_shape_features):
    feature_names = []
    feature_sizes = []

    if use_raw_pixels:
        feature_names.append("raw_pixels")
        feature_sizes.append(np.prod(image_size))
    if use_color_histogram:
        feature_names.append("color_histogram")
        feature_sizes.append(256)
    if use_edge_features:
        feature_names.append("edge_features")
        feature_sizes.append(np.prod(image_size))
    if use_lbp:
        feature_names.append("lbp_features")
        feature_sizes.append(10)
    if use_haralick:
        feature_names.append("haralick_features")
        feature_sizes.append(13)
    if use_gabor:
        feature_names.append("gabor_features")
        feature_sizes.append(50)
    if use_hog:
        feature_names.append("hog_features")
        feature_sizes.append(36)
    if use_color_moments:
        feature_names.append("color_moments")
        feature_sizes.append(9)
    if use_dominant_colors:
        feature_names.append("dominant_colors")
        feature_sizes.append(9)  # Assuming 3 dominant colors
    if use_stripe_spot:
        feature_names.append("stripe_spot_features")
        feature_sizes.append(1)
    if use_gradient:
        feature_names.append("gradient_features")
        feature_sizes.append(4)
    if use_fourier_descriptors:
        feature_names.append("fourier_descriptors")
        feature_sizes.append(256)  # Adjust size based on your implementation

    # New features
    if use_zernike_moments:
        feature_names.append("zernike_moments")
        feature_sizes.append(21)  # Zernike moments (adjust size as needed)
    if use_aspect_ratio:
        feature_names.append("aspect_ratio")
        feature_sizes.append(1)
    if use_shape_features:
        feature_names.append("shape_features")
        feature_sizes.append(2)  # Solidity and Circularity

    return feature_names, feature_sizes


# Parallelized function to process individual images
def process_image(image_path, image_size, feature_flags):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, image_size)
    features = []

    # Extract features based on flags
    if feature_flags['use_raw_pixels']:
        features.append(extract_raw_pixels(img_resized))
    if feature_flags['use_color_histogram']:
        features.append(extract_color_histogram(img_resized))
    if feature_flags['use_edge_features']:
        features.append(extract_edge_features(img_resized))
    if feature_flags['use_lbp']:
        features.append(extract_lbp_features(img_resized))
    if feature_flags['use_haralick']:
        features.append(extract_haralick_features(img_resized))
    if feature_flags['use_gabor']:
        features.append(extract_gabor_features(img_resized))
    if feature_flags['use_hog']:
        features.append(extract_hog_features(img_resized))
    if feature_flags.get('use_color_moments', False):
        features.append(extract_color_moments(img_resized))
    if feature_flags.get('use_dominant_colors', False):
        features.append(extract_dominant_colors(img_resized))
    if feature_flags.get('use_stripe_spot', False):
        features.append(extract_stripe_spot_features(img_resized))
    if feature_flags.get('use_gradient_features', False):
        features.append(extract_gradient_features(img_resized))
    if feature_flags.get('use_fourier_descriptors', False):
        features.append(extract_fourier_descriptors(img_resized))
    if feature_flags.get('use_zernike_moments', False):
        features.append(extract_zernike_moments(img_resized))

    # Concatenate all features into one array
    combined_features = np.concatenate(features)

    return combined_features


# Main function with parallelization
def load_images_and_apply_pca(folders, image_size=(128, 128), batch_size=500, n_components=10,
                              use_raw_pixels=False, use_color_histogram=False, use_edge_features=False,
                              use_lbp=False, use_haralick=False, use_gabor=False, use_hog=False,
                              use_color_moments=False, use_dominant_colors=False, use_stripe_spot=False,
                              use_gradient_features=False, use_fourier_descriptors=False, use_zernike_moments=False,
                              use_aspect_ratio=False, use_shape_features=False):
    """
    Load images in batches, extract selected features in parallel, and perform Incremental PCA.

    Returns:
        all_reduced_features (numpy.ndarray): PCA-reduced feature matrix.
        all_labels (numpy.ndarray): Array of labels corresponding to features.
        feature_names (list): Names of selected features.
        feature_sizes (list): Number of dimensions for each feature.
    """
    ipca = IncrementalPCA(n_components=n_components)
    all_labels = []
    all_reduced_features = []
    total_batches = 0

    # Get feature names and sizes
    feature_names, feature_sizes = get_feature_info(
        image_size,
        use_raw_pixels,
        use_color_histogram,
        use_edge_features,
        use_lbp,
        use_haralick,
        use_gabor,
        use_hog,
        use_color_moments,
        use_dominant_colors,
        use_stripe_spot,
        use_gradient_features,
        use_fourier_descriptors,
        use_zernike_moments,
        use_aspect_ratio,
        use_shape_features
    )

    # Calculate total number of batches
    for folder in folders:
        files = [f for f in os.listdir(folder) if f.endswith(".jpeg")]
        total_batches += (len(files) + batch_size - 1) // batch_size

    batch_count = 0
    batch_times = []  # Track time per batch

    print(f"Starting PCA with {total_batches} total batches...")

    for folder in folders:
        files = [f for f in os.listdir(folder) if f.endswith(".jpeg")]

        for i in range(0, len(files), batch_size):
            batch_start_time = time.time()
            batch_files = files[i:i + batch_size]
            X, y = [], []  # Reset features and labels for the current batch

            # Extract labels
            for filename in batch_files:
                parts = filename.split(" ")
                if len(parts) > 1 and "aug" not in parts[-1]:
                    label = " ".join(parts[:-1])
                elif len(parts) > 1 and "aug" in parts[-1]:
                    label = " ".join(parts[:-2])
                else:
                    label = parts[0]

                y.append(label)

            # Parallel feature extraction
            feature_flags = {
                'use_raw_pixels': use_raw_pixels,
                'use_color_histogram': use_color_histogram,
                'use_edge_features': use_edge_features,
                'use_lbp': use_lbp,
                'use_haralick': use_haralick,
                'use_gabor': use_gabor,
                'use_hog': use_hog,
                'use_color_moments': use_color_moments,
                'use_dominant_colors': use_dominant_colors,
                'use_stripe_spot': use_stripe_spot,
                'use_gradient_features': use_gradient_features,
                'use_fourier_descriptors': use_fourier_descriptors,
                'use_zernike_moments': use_zernike_moments,
                'use_aspect_ratio': use_aspect_ratio,
                'use_shape_features': use_shape_features
            }

            with ThreadPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(lambda path: process_image(
                    os.path.join(folder, path), image_size, feature_flags), batch_files))


            # Convert all features to NumPy arrays before filtering
            results = [np.asarray(r, dtype=np.float32) if r is not None else None for r in results]

            # Remove failed extractions and ensure all features have the same shape
            results = [r for r in results if r is not None and r.shape == results[0].shape]

            if len(results) == 0:
                print(f"Warning: No valid features extracted in batch {batch_count}. Skipping batch...")
                continue

            valid_results = [(r, label) for r, label in zip(results, y) if r is not None]
            if len(valid_results) == 0:
                print(f"Warning: No valid features extracted in batch {batch_count}. Skipping batch...")
                continue

            X, y = zip(*valid_results)
            X = np.array(X, dtype=np.float32)
            y = np.array(y)

            # Normalize the features in the current batch
            scaler = StandardScaler()  # Create an instance of StandardScaler
            X_normalized = scaler.fit_transform(X)  # Normalize features before PCA

            # Merging last small batch with the previous batch
            if i + batch_size >= len(files) and X_normalized.shape[0] < n_components:
                print(f"Merging last batch ({X_normalized.shape[0]} samples) with previous batch.")
                all_reduced_features.extend(ipca.transform(X_normalized))  # Transform & store
                all_labels.extend(y)
                continue  # Skip processing this batch alone

            # Fit Incremental PCA
            ipca.partial_fit(X_normalized)

            # Transform the current batch
            reduced_batch = ipca.transform(X_normalized)
            all_reduced_features.extend(reduced_batch)
            all_labels.extend(y)

            # Time estimation
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - batch_count
            estimated_remaining_time = remaining_batches * avg_batch_time

            batch_count += 1
            print(f"Batch {batch_count}/{total_batches} processed in {batch_time:.2f}s.")

            # Convert estimated remaining time to minutes and seconds
            remaining_minutes = int(estimated_remaining_time // 60)
            remaining_seconds = int(estimated_remaining_time % 60)

            print(f"Estimated time remaining: {remaining_minutes} minutes {remaining_seconds} seconds.")

    # Finalize PCA results
    print("All batches processed. PCA transformation complete.")
    all_reduced_features = np.array(all_reduced_features, dtype=np.float32)
    all_labels = np.array(all_labels)
    groups = all_labels
    joblib.dump(ipca, 'sklearn_model/pca_transformer.joblib')

    # Save PCA components and explained variance
    joblib.dump(ipca.components_, "sklearn_model/pca_components.joblib")
    joblib.dump(ipca.explained_variance_ratio_, "sklearn_model/explained_variance.joblib")

    return all_reduced_features, all_labels, feature_names, feature_sizes, groups



def calculate_feature_importance(components, explained_variance, feature_names, feature_sizes):
    """
    Calculate feature importance after PCA by aggregating contributions to components.

    Args:
        components (ndarray): PCA components (n_components x original_feature_dim).
        explained_variance (ndarray): Explained variance ratio for each PCA component.
        feature_names (list): List of feature groups (e.g., "raw_pixels", "color_histogram").
        feature_sizes (list): Number of dimensions for each feature group.

    Returns:
        dict: Feature importance for each feature group.
    """
    importance = np.zeros(len(feature_names))
    start_idx = 0

    for i, (feature_name, feature_size) in enumerate(zip(feature_names, feature_sizes)):
        end_idx = start_idx + feature_size
        group_contribution = np.sum(
            np.abs(components[:, start_idx:end_idx]) * explained_variance[:, np.newaxis], axis=0
        )
        importance[i] = np.sum(group_contribution)
        start_idx = end_idx

    # Normalize to sum to 1
    importance /= np.sum(importance)
    return dict(zip(feature_names, importance))


folder_path = "train"
num_images1 = count_images_in_folder(folder_path)
print(
    f"There are {num_images1} images, split into {num_images1 / 41} images in each of the 41 classes, in the folder '{folder_path}'.")
folder_path = "train_augmented"
# num_images2 = count_images_in_folder(folder_path)
# print(f"There are {num_images2} images in the folder '{folder_path}'.")
print(f"There are {num_images1 * 10} total images in the dataset")
print(f"Totaling {(num_images1 * 10) / 41} images per class")

# Specify folders and image size
folders = [
    "train/",
    "train_augmented",
    "train_2",
    "confusion"
]

image_size = (128, 128)
num_components = 150
# Check for pre-processed data
if os.path.exists('sklearn_model/processed_data.joblib'):
    # Load pre-processed data
    X, y, feature_names, feature_sizes, groups = joblib.load('sklearn_model/processed_data.joblib')
    print("Using pre-processed data")
else:
    # Load images and apply PCA while extracting features
    X, y, feature_names, feature_sizes, groups = load_images_and_apply_pca(
        folders,
        image_size=image_size,
        n_components=num_components,  # Adjust the number of components if needed
        use_raw_pixels=False,
        use_color_histogram=True,
        use_edge_features=False,
        use_lbp=True,
        use_haralick=True,
        use_gabor=True,
        use_hog=True,
        use_color_moments=False,
        use_dominant_colors=False,
        use_stripe_spot=False,
        use_gradient_features=False,
        use_fourier_descriptors=False,
        use_zernike_moments=True,
        use_aspect_ratio=False,
        use_shape_features=False
    )

    # Save processed data along with feature metadata
    joblib.dump((X, y, feature_names, feature_sizes, groups), 'sklearn_model/processed_data.joblib')
    print("Pre-processed data saved.")

    print(f"Starting PCA with {num_components} components")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'sklearn_model/label_encoder.joblib')
print("Images loaded and labels encoded.")

print(f"Final lengths: X={len(X)}, y={len(y)}, groups={len(groups)}")
assert len(X) == len(y) == len(groups), "Mismatch detected!"

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, stratify=y, random_state=42
)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Load PCA components and explained variance
components = joblib.load("sklearn_model/pca_components.joblib")
explained_variance = joblib.load("sklearn_model/explained_variance.joblib")

# Calculate feature importances
feature_importances = calculate_feature_importance(
    components, explained_variance, feature_names, feature_sizes
)

# Save feature importances for future use
joblib.dump(feature_importances, 'sklearn_model/feature_importances.joblib')
print("Feature importances saved.")

# Display feature importances
print("Feature Importances:")
for feature, importance in feature_importances.items():
    print(f"{feature}: {importance:.4f}")

# Sort features by importance in descending order
sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features)

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(features, importances, color='skyblue')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importances (Sorted)', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('Feature Imporances.jpeg')
# Show the plot
plt.show()

pca = joblib.load('sklearn_model/pca_transformer.joblib')

umap_model = umap.UMAP(n_neighbors=30, min_dist=0.5, n_jobs=1)
X_umap = umap_model.fit_transform(X_train)

umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
umap_df['Species'] = y_train  # Add species labels for coloring

# # Plot with Plotly
fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', color='Species',
                 title="UMAP Projection of Fish Species",
                 labels={'Species': 'Fish Species'})

# Show the plot
fig.update_layout(title="UMAP Projection of Fish Species")
fig.show()
