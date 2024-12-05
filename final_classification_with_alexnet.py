import torch
import torchvision.models as models
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet_model = models.alexnet(weights=None)
alexnet_model.classifier[6] = torch.nn.Linear(4096, 1)  
alexnet_model.to(device)
alexnet_model.load_state_dict(torch.load("alexnet_water_quality_classification.pth", map_location=device))
alexnet_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_image_path = "F:/GangaFlow/dataset/test/images/test2.jpg"

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at {test_image_path}")

image = cv2.imread(test_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
image_pil = Image.fromarray(image)  
image_transformed = transform(image_pil).unsqueeze(0).to(device)  
with torch.no_grad():
    output = alexnet_model(image_transformed)
    probability = torch.sigmoid(output).item()
if probability > 0.5:
    print(f"Prediction: Polluted (Probability: {probability:.4f})")
else:
    print(f"Prediction: Non-Polluted (Probability: {1 - probability:.4f})")
