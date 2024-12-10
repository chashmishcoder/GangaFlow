import torch
import cv2
import os
from ultralytics import YOLO
from segmentation_models_pytorch import Unet
from torchvision import models, transforms
import numpy as np
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")  

# Load U-Net Model
unet_model = Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation=None).to(device)
unet_model.load_state_dict(torch.load("unet_ganga_water_quality.pth", map_location=device))
unet_model.eval()

# Load AlexNet Model
alexnet_model = models.alexnet(pretrained=False).to(device)
alexnet_model.classifier[6] = torch.nn.Linear(4096, 1)  
alexnet_model.load_state_dict(torch.load("alexnet_water_quality_classification.pth", map_location=device))

# Now modify the output layer for 2 classes (your new requirement)
alexnet_model.classifier[6] = torch.nn.Linear(4096, 2)  # For 2 classes
alexnet_model.eval()

# Define Image Transform for AlexNet
alexnet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Process YOLO Predictions
def process_yolo_predictions(image, results, conf_threshold=0.25):
    detected_regions = []
    bboxes = []
    if results[0].boxes is not None:  # If objects are detected
        for box in results[0].boxes:
            conf = box.conf[0].item()  # Get the confidence score for the detection
            if conf >= conf_threshold:  # Only accept detections above the threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_regions.append(image[y1:y2, x1:x2])
                bboxes.append((x1, y1, x2, y2))
    return detected_regions, bboxes

# Process U-Net Model
def segment_region(region):
    region_resized = cv2.resize(region, (256, 256))
    region_tensor = torch.from_numpy(region_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        mask = unet_model(region_tensor)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()
    # Return the mask as a binary mask (0 or 255)
    return (mask > 0.5).astype(np.uint8) * 255  # Multiply by 255 to get proper 8-bit mask

# Classify Region with AlexNet
def classify_region(region):
    region_pil = Image.fromarray(region)
    region_transformed = alexnet_transform(region_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = alexnet_model(region_transformed)
        _, prediction = torch.max(output, 1)
    return "Polluted" if prediction.item() == 1 else "Non-Polluted"

# Calculate Pollution Percentage
def calculate_pollution_percentage(mask):
    # Calculate the percentage of "polluted" area in the mask
    polluted_area = np.sum(mask == 255)  # Count the number of polluted pixels (255)
    total_area = mask.size  # Total number of pixels
    pollution_percentage = (polluted_area / total_area) * 100
    return pollution_percentage

# Testing Pipeline
test_images_dir = "F:/GangaFlow/dataset/test/images"  # Test images folder
output_dir = "F:/GangaFlow/output_test"  # Output directory for results
os.makedirs(output_dir, exist_ok=True)

# List test images
test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))] 

# Debug print statement to check number of images
print(f"Found {len(test_images)} test images.")

# Loop through each image and process
for img_path in test_images:
    print(f"Processing image: {img_path}")
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: YOLO Detection
    yolo_results = yolo_model(image_rgb)
    detected_regions, bboxes = process_yolo_predictions(image, yolo_results, conf_threshold=0.25)  

    if not detected_regions:
        print("No detections found in this image.")  # Debugging: print if no detections
        continue

    # Prepare a list to store the masked regions for the current image
    region_outputs = []

    for idx, (region, bbox) in enumerate(zip(detected_regions, bboxes)):
        # Step 2: U-Net Segmentation
        mask = segment_region(region)

        # Ensure the mask is the correct size and type
        mask_resized = cv2.resize(mask, (region.shape[1], region.shape[0]))  # Resize mask to match region size
        mask_resized = np.uint8(mask_resized)  # Ensure it's an 8-bit single-channel mask

        # Step 3: Apply mask to the region
        masked_region = cv2.bitwise_and(region, region, mask=mask_resized)

        # Step 4: Classification with AlexNet
        classification_result = classify_region(masked_region)
        print(f"Region {idx + 1}: {classification_result}")  # Debugging: print classification result

        # Step 5: Calculate pollution percentage
        pollution_percentage = calculate_pollution_percentage(mask_resized)

        # Draw bounding box and label on the original image
        x1, y1, x2, y2 = bbox
        label = f"{classification_result} ({pollution_percentage:.2f}% Polluted)"
        color = (0, 255, 0) if classification_result == "Non-Polluted" else (0, 0, 255)  # Green for Non-Polluted, Red for Polluted
        thickness = 2

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Put the label with a background box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Step 6: Save the image with bounding boxes and segmentation mask
    base_name = os.path.basename(img_path)
    output_filename = f"{base_name}_processed.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image)

    print(f"Saved output image: {output_path}")

print("Testing complete. Results saved.")
