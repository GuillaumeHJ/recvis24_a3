import torchvision.transforms as transforms

data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


