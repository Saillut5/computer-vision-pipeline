import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        if pretrained:
            self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Freeze all parameters in the feature extractor
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False
            # Replace the classifier head
            in_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.efficientnet = efficientnet_b0(weights=None)
            in_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.efficientnet(x)


if __name__ == '__main__':
    # Example usage:
    # Test with pretrained EfficientNet
    model_pretrained = EfficientNetClassifier(num_classes=10, pretrained=True)
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    output_pretrained = model_pretrained(input_tensor)
    print(f"Pretrained EfficientNet output shape: {output_pretrained.shape}")
    assert output_pretrained.shape == (1, 10), f"Expected (1, 10), got {output_pretrained.shape}"

    # Test without pretrained EfficientNet
    model_scratch = EfficientNetClassifier(num_classes=5, pretrained=False)
    input_tensor_scratch = torch.randn(1, 3, 224, 224)
    output_scratch = model_scratch(input_tensor_scratch)
    print(f"Scratch EfficientNet output shape: {output_scratch.shape}")
    assert output_scratch.shape == (1, 5), f"Expected (1, 5), got {output_scratch.shape}"

    print("All EfficientNet tests passed!")

    # Verify that feature extractor parameters are frozen for pretrained model
    frozen_params = 0
    total_params = 0
    for name, param in model_pretrained.named_parameters():
        total_params += 1
        if not param.requires_grad:
            frozen_params += 1
    print(f"Total parameters in pretrained model: {total_params}")
    print(f"Frozen parameters in pretrained model: {frozen_params}")
    # Expecting most of the feature extractor parameters to be frozen
    assert frozen_params > 50, "Feature extractor parameters not frozen as expected."
# Simulated change on 2023-01-05 13:54:00
