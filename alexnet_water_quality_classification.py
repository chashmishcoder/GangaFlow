import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

image_folder = "F:/Gangaflow/dataset/train/images" 
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
labels = [1 if 'polluted' in f else 0 for f in os.listdir(image_folder)]  

dataset = ClassificationDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = models.alexnet(pretrained=True)


model.classifier[6] = nn.Linear(4096, 1)  
model = model.to(device)


criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-4)


model_name = "alexnet"
output_folder = os.path.join("F:/Gangaflow/output_visualization_classification", model_name)
os.makedirs(output_folder, exist_ok=True)

epochs = 10  
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)  

        # Forward pass
        preds = model(imgs)
        loss = criterion(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), f"{model_name}_water_quality_classification.pth")
print("Model saved successfully!")


with torch.no_grad():
    model.eval()
    
    sample_images = [image_paths[i] for i in range(4)]  
    sample_labels = [labels[i] for i in range(4)]     

    for i, (image_path, label) in enumerate(zip(sample_images, sample_labels)):
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  

    
        pred = model(img_tensor)
        pred_label = torch.sigmoid(pred).squeeze().cpu().item()  
        pred_label = 1 if pred_label >= 0.5 else 0  
      
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"True: {'Polluted' if label == 1 else 'Non-polluted'} | Pred: {'Polluted' if pred_label == 1 else 'Non-polluted'}")
        plt.axis('off')


        img_filename = os.path.join(output_folder, f"sample_{i+1}_result.png")
        plt.savefig(img_filename)
        plt.close()

        print(f"Saved visual result for sample {i+1}")
