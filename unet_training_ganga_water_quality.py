import torch
import cv2
import os
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        mask = mask / 255.0

        return img, mask.unsqueeze(0)  

image_folder = "F:/Gangaflow/dataset/train/images"  
mask_folder = "F:/Gangaflow/dataset/train/masks"    

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder)]
transform = A.Compose([
    A.Resize(256, 256),  # Resize images
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.Normalize(),  # Normalize pixel values
    ToTensorV2(),   # Convert to PyTorch tensors
])


dataset = SegmentationDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None).to(device)

criterion = torch.nn.BCEWithLogitsLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  

output_folder = "F:/Gangaflow/output_visualization"
os.makedirs(output_folder, exist_ok=True)
epochs = 10  
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

        imgs, masks = imgs.to(device), masks.to(device)

        # Forward pass
        preds = model(imgs)
        loss = criterion(preds, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "unet_ganga_water_quality.pth")
print("Model saved successfully!")
with torch.no_grad():
    model.eval()
    sample_images = [image_paths[i] for i in range(4)]  
    sample_masks = [mask_paths[i] for i in range(4)]   
    
    for i, (image_path, mask_path) in enumerate(zip(sample_images, sample_masks)):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(device)  # (1, 3, H, W)
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
        pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()  # Convert to binary mask
        pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0])) * 255
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0])) * 255
        img_filename = os.path.join(output_folder, f"final_img_{i+1}.png")
        mask_filename = os.path.join(output_folder, f"final_gt_mask_{i+1}.png")
        pred_filename = os.path.join(output_folder, f"final_pred_mask_{i+1}.png")

        cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  
        cv2.imwrite(mask_filename, mask_resized)  
        cv2.imwrite(pred_filename, pred_mask_resized)  

        print(f"Saved final images and masks for sample {i+1}")
