import torchvision.transforms as transforms
import torch as torch

data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamp values to [0, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# data_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


