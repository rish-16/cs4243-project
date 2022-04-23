import torch
import torch.nn as nn
from torchvision import transforms

def get_CAM(feature_map, weight, class_idx):
    size_upsample = (32, 32)
    bz, nc, h, w = feature_map.shape

    before_dot = feature_map.reshape((nc, h*w))
    cam = weight[class_idx].unsqueeze(0) @ before_dot

    cam = cam.squeeze(0)
    cam = cam.reshape(h, w)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    cam = torch.clip(cam, 0, 1)
    
    img = transforms.Resize(size_upsample)(cam.unsqueeze(0))
    
    return img.detach().numpy(), cam

def plot_cam(img, cam, ALLCLASSES):
    ''' 
    Visualization function

    `ALLCLASSES` is a simple list of class label strings. CIFAR-10 example usage:

    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    img, cam = ...
    plot_cam(img, cam, cifar10_classes)
    '''

    img = img.permute(1, 2, 0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,7))
    ax1.imshow(img)
    ax1.set_title(f"Input image\nLabel: {ALLCLASSES[y]}")

    ax2.imshow(cam.reshape(32, 32), cmap="jet")
    ax2.set_title("Raw CAM.")

    ax3.imshow(img)
    ax3.imshow(cam.reshape(32, 32), cmap="jet", alpha=0.2)
    ax3.set_title(f"Overlayed CAM.\nPrediction: {ALLCLASSES[idx[0]]}")
    plt.show()