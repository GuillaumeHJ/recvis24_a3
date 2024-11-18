import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import timm  

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class NASNetMobile(nn.Module):
    def __init__(self, freeze_layers=True):
        super().__init__()  # Just use super() without any arguments in modern Python
        self.base_model = timm.create_model('mnasnet_100', pretrained=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, nclasses)

        if freeze_layers:
            self.freeze_base_layers()

    def freeze_base_layers(self):
        """Freeze the base layers to update only the final classifier."""
        for name, param in self.base_model.named_parameters():
            if 'classifier' not in name:  # Freeze all layers except the classifier
                param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)


class DINOv2(nn.Module):
    def __init__(self, freeze_layers=True):
        super(DINOv2, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.base_model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-base')
        self.fc = nn.Linear(self.base_model.config.hidden_size, nclasses)
        
        if freeze_layers:
            self.freeze_base_layers()

    def freeze_base_layers(self):
        """Freeze the base model layers to train only the final classifier."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        xinputs = self.processor(images=x, return_tensors="pt").pixel_values
        hidden_states = self.base_model(xinputs)[0][:, 0, :]  # Use CLS token representation
        return self.fc(hidden_states)


class EfficientNetB3a(nn.Module):
    def __init__(self, freeze_layers=True):
        super(EfficientNetB3a, self).__init__()
        self.base_model = timm.create_model('efficientnet_b3a', pretrained=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, nclasses)

        if freeze_layers:
            self.freeze_base_layers()

    def freeze_base_layers(self):
        """Freeze the base layers to update only the final classifier."""
        for name, param in self.base_model.named_parameters():
            if 'classifier' not in name:  # Freeze all layers except the classifier
                param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)

