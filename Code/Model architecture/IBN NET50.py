import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

# Define the IBN-Net50-a architecture
class IBNNet50a(ResNet):
    def __init__(self, num_classes=1000):
        # Define the Bottleneck layers with IBN
        block = Bottleneck
        layers = [3, 4, 6, 3]  # Standard ResNet50 layers
        super(IBNNet50a, self).__init__(block, layers)

        # Modify the fully connected layer to match the number of classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

model_IBNNET = IBNNet50a(num_classes=1501) # 1501 is the number of classes in Market-1501
print(model_IBNNET)

