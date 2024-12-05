import torch
import torchvision.models as models
from ultralytics import YOLO
import cv2
import os
import numpy as np
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("yolov8n.pt") 
yolo_model.to(device)

unet_model = Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation=None).to(device)
unet_model.load_state_dict(torch.load("unet_ganga_water_quality.pth", map_location=device))
unet_model.eval()

alexnet_model = models.alexnet(pretrained=True)
alexnet_model.classifier[6] = torch.nn.Linear(4096, 1)  
alexnet_model.load_state_dict(torch.load("alexnet_water_quality_classification.pth", map_location=device))
alexnet_model.to(device)
alexnet_model.eval()

unet_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

alexnet_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])
output_folder = "F:/GangaFlow/integration_outputs"
os.makedirs(output_folder, exist_ok=True)
def process_yolo_predictions(image, results):
    cropped_regions = []
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cropped_regions.append(cropped)
        boxes.append((x1, y1, x2, y2))
    return cropped_regions, boxes

def apply_unet_on_region(region):
    transformed = unet_transform(image=region)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        mask = unet_model(img_tensor)
        mask = mask.squeeze().cpu().numpy()
    return mask

def classify_region_with_alexnet(region):
    transformed = alexnet_transform(image=region)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = alexnet_model(img_tensor)
        prob = torch.sigmoid(output).item()
    return "Polluted" if prob >= 0.5 else "Non-Polluted", prob

test_image_path = "F:/GangaFlow/dataset/test/images/test1.jpg"

image = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

yolo_results = yolo_model.predict(image_rgb, device=device, verbose=False)
if len(yolo_results[0].boxes) == 0:
    print("No detections found.")
    exit()
regions, boxes = process_yolo_predictions(image_rgb, yolo_results[0])

output_log = []
for i, (region, box) in enumerate(zip(regions, boxes)):
    x1, y1, x2, y2 = box
    mask = apply_unet_on_region(region)
    mask = (mask > 0.5).astype(np.uint8) * 255
    classification, prob = classify_region_with_alexnet(region)


    region_output_path = os.path.join(output_folder, f"region_{i+1}.jpg")
    mask_output_path = os.path.join(output_folder, f"mask_{i+1}.png")
    cv2.imwrite(region_output_path, cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_output_path, mask)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{classification} ({prob:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    output_log.append(f"Region {i + 1}: {classification} ({prob:.2f}) | Bounding Box: {box}")

annotated_image_path = os.path.join(output_folder, "annotated_image.jpg")
cv2.imwrite(annotated_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
log_file = os.path.join(output_folder, "results_log.txt")
with open(log_file, "w") as f:
    for line in output_log:
        f.write(line + "\n")

print("Integration and output generation complete!")
