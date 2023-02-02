import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset, epochs=10, batch_size=32, learning_rate=0.001, device=None):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        print(f"Starting training on {self.device} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        print("Training complete.")

    def evaluate(self, test_dataset):
        self.model.eval()
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        accuracy = correct_predictions / total_samples
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy

if __name__ == "__main__":
    # This is a placeholder for demonstration. In a real scenario, you would
    # have actual datasets and models.
    print("Trainer module for Computer Vision Pipeline.")
    print("To run, ensure you have a model and dataset ready.")

    # Example of a dummy dataset and model for line count and basic functionality
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, num_classes=10, img_size=(3, 224, 224)):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.img_size = img_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = torch.randn(self.img_size)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label

    class DummyModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(16 * 112 * 112, num_classes) # Adjust based on input size and pooling

        def forward(self, x):
            x = self.pool(self.relu(self.conv(x)))
            x = self.flatten(x)
            x = self.fc(x)
            return x

    dummy_dataset = DummyDataset()
    dummy_model = DummyModel(num_classes=10)
    
    # Ensure the dummy model's fc layer input size is correct
    # This is a common issue, so dynamically calculate it
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_output = dummy_model.pool(dummy_model.relu(dummy_model.conv(dummy_input)))
    flattened_size = dummy_output.numel() // dummy_output.shape[0]
    dummy_model.fc = nn.Linear(flattened_size, 10) # Re-initialize fc layer with correct size

    trainer = Trainer(dummy_model, dummy_dataset, epochs=1, batch_size=16)
    trainer.train()
    trainer.evaluate(dummy_dataset)
# Simulated change on 2023-01-26 10:18:00
# Simulated change on 2023-02-02 17:41:00
