import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
import numpy as np

# the path of input training seen images in a scene
input_folder = 'sr'

# image preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

def images_concat(image, kelnel_size=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kelnel_size, kelnel_size), sigma)

features_tensor = []

for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG'):
        img_path = os.path.join(input_folder, filename)
        img_tensor = cv2.imread(img_path)

        img_tensor = img_tensor/255.0

        filter_image = images_concat(img_tensor)
        output = img_tensor - filter_image
        features_tensor.append(output)

features_tensor = torch.stack([torch.from_numpy(arr) for arr in features_tensor], dim=0)
features_tensor = features_tensor.view(30, 3, 504 * 378)
features_tensor = features_tensor.float()
features_tensor = F.adaptive_avg_pool1d(features_tensor, 1000).permute(0, 2, 1)

# the path of saved images
torch.save(features_tensor, "rgb_sr.pth")




