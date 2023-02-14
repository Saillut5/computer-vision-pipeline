import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Example usage:
    # Create dummy data for demonstration
    os.makedirs("data/images", exist_ok=True)
    Image.new("RGB", (64, 64), color = 'red').save("data/images/img1.png")
    Image.new("RGB", (64, 64), color = 'blue').save("data/images/img2.png")
    pd.DataFrame({"filename": ["img1.png", "img2.png"], "label": [0, 1]}).to_csv("data/labels.csv", index=False)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_dir="data/images", labels_file="data/labels.csv", transform=transform)
    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
# Simulated change on 2023-01-16 18:52:00
# Simulated change on 2023-01-17 18:58:00
# Simulated change on 2023-01-30 12:58:00
# Simulated change on 2023-02-07 12:12:00
# Simulated change on 2023-02-14 18:36:00
