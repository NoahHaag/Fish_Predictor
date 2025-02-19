import math
from collections import Counter

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from compel import Compel
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


# from Deep learning model.py import FishClassifierCNN


def best_grid_shape(N):
    """
    Determine the best grid shape for displaying N images.
    Args:
        N (int): The total number of images (must be even).
    Returns:
        tuple: A tuple (rows, columns) representing the best grid shape.
    """
    if N <= 0 or N % 2 != 0:
        raise ValueError("N must be a positive even number.")

    # Start with the square root of N
    root = int(math.sqrt(N))

    # Find the closest factor pair (rows, columns)
    for rows in range(root, 0, -1):
        if N % rows == 0:
            columns = N // rows
            return rows, columns





def aggregate_predictions(probabilities, label_encoder):
    """
    Aggregates predictions from multiple generated images using Majority Voting and Mean Probabilities.

    Args:
        probabilities (np.ndarray): Array of shape (num_images, num_classes) with class probabilities.
        label_encoder: Trained label encoder to convert class indices back to labels.

    Returns:
        dict: Contains final predicted class, mean confidence, and majority vote results.
    """

    #  Majority Voting**
    predicted_classes = [label_encoder.inverse_transform([np.argmax(probs)])[0] for probs in probabilities]
    most_common_class, vote_count = Counter(predicted_classes).most_common(1)[0]

    #  Mean Probabilities**
    mean_probabilities = np.mean(probabilities, axis=0)
    final_class_index = np.argmax(mean_probabilities)
    final_class = label_encoder.inverse_transform([final_class_index])[0]
    final_confidence = mean_probabilities[final_class_index] * 100  # Convert to percentage

    # **Print Results**
    print(f"\nMajority Voting: {most_common_class} (Votes: {vote_count}/{len(probabilities)})")
    print(f"Mean Probabilities: {final_class}, Confidence: {final_confidence:.2f}%")

    return {
        "majority_vote_class": most_common_class,
        "majority_vote_count": vote_count,
        "final_class": final_class,
        "final_confidence": final_confidence
    }


# output_dir = "test/"
image_size = (128, 128)

main_color = "yellow with black vertical stripes"
secondary_color = "black stripes on a bright yellow body"
body_shape = "slightly flattened, kind of rectangular"
num_v_stripes = 6  # Black vertical stripes
num_h_stripes = 0  # No horizontal stripes
num_spots = ""  # No spots
mouth_angle = "neutral, not too pointed"
habitat = "swimming around coral and rocks, usually near the shore"
behavioral_traits = "they were swimming in small groups, a bit curious and not too shy"
group_behavior = "often in large schools, swimming near the reef, darting in and out of the rocks"

# Handle optional descriptions
behavioral_description = f" It exhibits {behavioral_traits} behavior traits." if behavioral_traits else ""
mouth_description = f"The fish's mouth is {mouth_angle}, adding to its unique appearance." if mouth_angle else ""
habitat_description = f"It is found in a {habitat} habitat." if habitat else ""

# Species description
species_description = f"A photorealistic image of a fish,"

# Group behavior description
group_behavior_description = f"It is known for {group_behavior}." if group_behavior else ""

if ["group_behavior"]:
    group_behavior = f"It is known for {group_behavior}. "

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
torch.backends.cudnn.benchmark = False

# Construct prompt
prompt = (
    f"{species_description} with {main_color} scales and a {body_shape} body shape. "
    f"The fish has {secondary_color} coloration, featuring {num_v_stripes} vertical stripes, "
    f"{num_h_stripes} horizontal stripes, and {num_spots} spots. {mouth_description} "
    f"{behavioral_description} {group_behavior_description} {habitat_description} "
    f"The fish is fully visible in the frame, showing its entire body, from head to tail, with all fins included. "
    f"It is perfectly centered in the image, displayed in a balanced and symmetrical composition. The focus is entirely on the fish, "
    f"capturing every detail of its scales, fins, eyes, and overall anatomy with lifelike textures. The background is minimal "
    f"and softly blurred to avoid distractions, with natural lighting highlighting the fish's features."
)

negative_prompt = (
    "cartoonish features, unrealistic textures, distorted anatomy, exaggerated proportions, missing fins, "
    "blurred or unclear details, incorrect colors, overly stylized elements, fantasy-like patterns, fake aquatic environments, "
    "abstract designs, artistic interpretations, non-fish features, humanoid traits, human-like skin tones, overly smooth or flesh-like textures, "
    "unnatural lighting, overexposed highlights, underexposed shadows, partial fish, cropped bodies, zoomed-in scales, unnatural poses, "
    "backgrounds with distracting elements, floating objects, symmetry errors, mechanical parts, artificial markings, textures or patterns inconsistent "
    "with real-world fish, species that do not resemble real-world aquatic life, depictions that could be mistaken for human or mammalian anatomy, "
    "poses or compositions that suggest ambiguity in the fish's nature, unnatural or overly suggestive shapes, and environments that do not clearly depict underwater settings."
)


# Load CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inception Module
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 3x3 convolution (after 1x1 reduction)
        self.conv3x3_reduce = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv3x3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 5x5 convolution (after 1x1 reduction)
        self.conv5x5_reduce = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.conv5x5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # Max pooling followed by 1x1 convolution
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1 = F.relu(self.conv1x1(x))

        branch2 = F.relu(self.conv3x3_reduce(x))
        branch2 = F.relu(self.conv3x3(branch2))

        branch3 = F.relu(self.conv5x5_reduce(x))
        branch3 = F.relu(self.conv5x5(branch3))

        branch4 = F.relu(self.pool_conv(self.pool(x)))

        # Concatenate along depth dimension
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Updated CNN with Inception Modules
class FishClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FishClassifierCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Replace second and third convolutional layers with Inception blocks
        self.inception1 = InceptionBlock(64)  # First Inception Block
        self.inception2 = InceptionBlock(320)  # Second Inception Block (concatenation increases depth)

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
            x = self.pool(self.inception1(x))
            x = self.pool(self.inception2(x))
            return x.view(1, -1).shape[1]  # Dynamically compute

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.inception1(x))
        x = self.pool(self.inception2(x))

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


# Manually set the number of classes (make sure it matches your trained model)
num_classes = 41  # Update this to match your dataset

# Load trained model
model = FishClassifierCNN(num_classes).to(device)
model.load_state_dict(torch.load("CNN models/fish_classifier.pth", weights_only=True))
model.eval()


# Process and predict user images
N = 10
prompt_embeds = compel_proc(prompt)

generated_images = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt=negative_prompt,
    guidance_scale=9,
    num_inference_steps=13,
    num_images_per_prompt=N
).images  # List of PIL images

class_names = joblib.load('class_names.joblib')

# Convert images to tensors and move to device
input_tensors = torch.stack([transform(img) for img in generated_images]).to(device)

# Perform inference
with torch.no_grad():
    outputs = model(input_tensors)
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()  # Convert to probabilities

# Aggregate and interpret predictions
total_confidence = 0
highest_confidence = 0
most_confident_class = None

for i, probs in enumerate(probabilities):
    class_index = np.argmax(probs)
    class_name = class_names[class_index]  # Get class name from list
    confidence = probs[class_index] * 100
    total_confidence += confidence

    if confidence > highest_confidence:
        highest_confidence = confidence
        most_confident_class = class_name

    print(f"Image {i + 1}: Predicted Class: {class_name}, Confidence: {confidence:.2f}%")

# Compute mean confidence
mean_confidence = total_confidence / len(probabilities)

print(f"\nMean Confidence: {mean_confidence:.2f}%")
print(f"Class with Highest Confidence: {most_confident_class}, Confidence: {highest_confidence:.2f}%")

# Compute final prediction based on averaged probabilities
mean_probabilities = np.mean(probabilities, axis=0)
final_class_index = np.argmax(mean_probabilities)
final_prediction = class_names[final_class_index]
final_confidence = mean_probabilities[final_class_index] * 100

print(f"\nFinal Predicted Species (Mean Probability): {final_prediction}, Confidence: {final_confidence:.2f}%")

# Confidence-weighted voting
weighted_votes = np.sum(probabilities, axis=0)
final_class_index = np.argmax(weighted_votes)
final_prediction = class_names[final_class_index]
final_confidence = weighted_votes[final_class_index] / np.sum(weighted_votes) * 100

print(f"\nFinal Predicted Species (Confidence-Weighted): {final_prediction}, Confidence: {final_confidence:.2f}%")