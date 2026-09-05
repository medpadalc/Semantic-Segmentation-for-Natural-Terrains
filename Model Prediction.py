import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class CustomModel_3(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.pretrained_model = deeplabv3_resnet50(weights=weights)
        self.one_layer = nn.Conv2d(21, 32, kernel_size, padding=2)
        self.two_layer = nn.Conv2d(32, 32, kernel_size, padding=2)
        self.final_layer = nn.Conv2d(32, 25, kernel_size, padding=2)

        self.one_batchnorm = nn.BatchNorm2d(32)
        self.two_batchnorm = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.pretrained_model(x)['out']

        y = self.one_layer(y)
        y = self.one_batchnorm(y)
        y = self.relu(y)

        y = self.two_layer(y)
        y = self.two_batchnorm(y)
        y = self.relu(y)

        return self.final_layer(y)



model = CustomModel_3(kernel_size=5)

# map_location ensures it loads correctly even if you trained on GPU 
# but are now running on a CPU-only machine
import os

def join_files(output_path, part_prefix, num_parts):
    with open(output_path, "wb") as outfile:
        for i in range(1, num_parts + 1):
            part_filename = f"{part_prefix}.part{i}"
            with open(part_filename, "rb") as infile:
                outfile.write(infile.read())

weights_path = "Model Weights/custom_model_3_weights.pth"
if not os.path.exists(weights_path):
    join_files(weights_path, "Model Weights/custom_model_3_weights.pth", num_parts=7)

state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.eval()

from PIL import Image
import torchvision.transforms as transforms
import torch

# Load the image
image_path = r"C:\Users\medpa\Downloads\Research Project\trail-10_00171.png"
image = Image.open(image_path).convert("RGB")

# Preprocess — MUST match whatever preprocessing was used during training
# (resize dims, normalization stats) or predictions will be garbage
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # match training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats — 
                          std=[0.229, 0.224, 0.225]),   # standard if using pretrained backbone
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # add batch dimension: [1, C, H, W]

# Run inference
with torch.no_grad():
    output = model(input_batch)
    # DeepLabv3 output is typically a dict with 'out' key
    if isinstance(output, dict):
        output = output['out']

# Get predicted class per pixel
pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

print("Prediction shape:", pred_mask.shape)
print("Unique classes predicted:", set(pred_mask.flatten()))




import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="tab20")  # tab20 gives distinct colors per class
plt.title("Predicted Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
