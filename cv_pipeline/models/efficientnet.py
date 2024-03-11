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
# Simulated change on 2023-01-18 13:54:00
# Simulated change on 2023-02-24 13:58:00
# Simulated change on 2023-03-03 12:49:00
# Simulated change on 2023-03-16 16:18:00
# Simulated change on 2023-03-21 10:56:00
# Simulated change on 2023-03-31 18:26:00
# Simulated change on 2023-04-05 12:36:00
# Simulated change on 2023-04-13 16:48:00
# Simulated change on 2023-04-13 11:14:00
# Simulated change on 2023-04-17 10:03:00
# Simulated change on 2023-05-02 13:23:00
# Simulated change on 2023-05-19 16:31:00
# Simulated change on 2023-06-23 09:08:00
# Simulated change on 2023-07-04 09:42:00
# Simulated change on 2023-07-06 18:46:00
# Simulated change on 2023-08-17 11:36:00
# Simulated change on 2023-09-21 17:15:00
# Simulated change on 2023-09-28 17:37:00
# Simulated change on 2023-10-05 12:57:00
# Simulated change on 2023-10-09 16:37:00
# Simulated change on 2023-10-27 09:09:00
# Simulated change on 2023-11-27 14:39:00
# Simulated change on 2023-12-08 14:56:00
# Simulated change on 2023-12-19 10:30:00
# Simulated change on 2023-12-26 10:26:00
# Simulated change on 2023-12-28 10:03:00
# Simulated change on 2024-01-05 13:28:00
# Simulated change on 2024-01-24 16:35:00
# Simulated change on 2024-02-09 11:53:00
# Simulated change on 2024-02-14 16:52:00
# Simulated change on 2024-02-19 18:00:00
# Simulated change on 2024-02-29 11:55:00
# Simulated change on 2024-03-05 18:29:00
# Simulated change on 2024-03-11 10:31:00
