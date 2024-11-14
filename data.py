import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random

# # once the images are loaded, how do we pre-process them before being passed into the network
# # by default, we resize the images to 64 x 64 in size
# # and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
# data_transforms = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )



# Define a custom function to apply Grayscale conversion
def grayscale(image):
    return transforms.Grayscale(num_output_channels=3)(image)

# Define a custom function to apply Median Blur using OpenCV
def median_blur(image):
    # Convert the image from PIL to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    # Apply median blur
    image_blurred = cv2.medianBlur(image_cv, 5)  # 5 is the kernel size
    # Convert back to PIL image
    image_blurred = Image.fromarray(cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB))
    return image_blurred

# Define a custom function to apply Laplacian Filter for edge detection
def laplacian_filter(image):
    # Convert the image from PIL to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image_cv, cv2.CV_64F)
    # Convert back to PIL image
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian = Image.fromarray(cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB))
    return laplacian

# Define a custom function for random cropping
def random_crop(image):
    width, height = image.size
    left = random.randint(0, width // 4)
    top = random.randint(0, height // 4)
    right = random.randint(width // 2, width)
    bottom = random.randint(height // 2, height)
    
    return image.crop((left, top, right, bottom))

# Data transforms with grayscale, median blur, Laplacian filter, and data augmentation
data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        
        # Grayscale conversion
        transforms.Lambda(lambda x: grayscale(x)),
        
        # Apply Median Blur
        transforms.Lambda(lambda x: median_blur(x)),
        
        # Apply Laplacian Filter
        transforms.Lambda(lambda x: laplacian_filter(x)),
        
        # Data Augmentation: Flipping and Rotating
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([90, 180, 270]),

        # Random Cropping
        transforms.Lambda(lambda x: random_crop(x)),
        
        # Convert to tensor and normalize (ImageNet stats)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
