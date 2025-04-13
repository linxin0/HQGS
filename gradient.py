import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F

# the path of the training seen images in a scene
input_folder = 'sr_seen'


# image preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Sobel operator
sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)

features_tensor = []

for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPG'):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        grads = []
        for channel in range(3):
            channel_tensor = img_tensor[channel].unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
            grad_x = torch.nn.functional.conv2d(channel_tensor, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(channel_tensor, sobel_y, padding=1)
            grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            grads.append(grad)

        grad_tensor = torch.cat(grads, dim=1).squeeze(0)  # shape: [3, H, W]

        grad = (1 - grad_tensor) * img_tensor
        features_tensor.append(grad)

features_tensor = torch.stack(features_tensor, dim=0)
# [the number of the training data, 3, H*W]
features_tensor = features_tensor.view(30, 3, 504 * 378)
features_tensor = F.adaptive_avg_pool1d(features_tensor, 1000).permute(0, 2, 1)

# saved path
torch.save(features_tensor, "sr.pth")

def gradient(image):
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).cuda()
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).cuda()

    image = image.clamp(0, 1) * 255
    image = image.to(torch.uint8)
    image = image.to(torch.float32)
    img_tensor = image

    grads = []
    for channel in range(3):
        channel_tensor = img_tensor[channel].unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        grad_x = torch.nn.functional.conv2d(channel_tensor, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(channel_tensor, sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grads.append(grad)


    grad_tensor = torch.cat(grads, dim=1).squeeze(0)  # shape: [3, H, W]
    min_grad = grad_tensor.min()
    max_grad = grad_tensor.max()
    normal_grad_tensor = (grad_tensor - min_grad) / (max_grad - min_grad)

    return normal_grad_tensor



