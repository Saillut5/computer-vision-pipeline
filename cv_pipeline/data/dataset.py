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
# Simulated change on 2023-02-20 15:31:00
# Simulated change on 2023-02-20 14:11:00
# Simulated change on 2023-03-08 11:11:00
# Simulated change on 2023-04-06 15:22:00
# Simulated change on 2023-04-12 10:51:00
# Simulated change on 2023-04-17 10:15:00
# Simulated change on 2023-04-28 15:24:00
# Simulated change on 2023-05-09 14:13:00
# Simulated change on 2023-05-11 14:36:00
# Simulated change on 2023-05-17 09:00:00
# Simulated change on 2023-05-23 16:29:00
# Simulated change on 2023-05-31 14:37:00
# Simulated change on 2023-06-07 12:27:00
# Simulated change on 2023-06-13 14:52:00
# Simulated change on 2023-07-06 13:00:00
# Simulated change on 2023-07-19 11:10:00
# Simulated change on 2023-08-29 14:35:00
# Simulated change on 2023-08-31 11:27:00
# Simulated change on 2023-09-01 09:49:00
# Simulated change on 2023-09-06 12:19:00
# Simulated change on 2023-10-16 10:58:00
# Simulated change on 2023-10-17 11:00:00
# Simulated change on 2023-10-25 10:07:00
# Simulated change on 2023-11-20 15:59:00
# Simulated change on 2023-11-27 17:31:00
# Simulated change on 2023-11-28 17:49:00
# Simulated change on 2023-12-25 14:12:00
# Simulated change on 2024-01-16 16:17:00
# Simulated change on 2024-02-09 15:23:00
# Simulated change on 2024-02-27 12:11:00
# Simulated change on 2024-03-05 11:07:00
# Simulated change on 2024-03-05 10:22:00
# Simulated change on 2024-03-07 14:42:00
# Simulated change on 2024-03-29 17:40:00
