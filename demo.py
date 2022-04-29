import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from engine import Engine
from model_training import *
from torchvision import transforms

def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class CNN(nn.Module):
    CHANNELS = [64, 128, 192, 256, 512]
    POOL = (1, 1)
    def __init__(self, n_channels, n_classes=9, dropout=0):
        super(CNN, self).__init__()
        layer1 = convbn(n_channels, self.CHANNELS[1], kernel_size=3, stride=2, padding=1, bias=True)
        layer2 = convbn(self.CHANNELS[1], self.CHANNELS[2], kernel_size=3, stride=2, padding=1, bias=True)
        layer3 = convbn(self.CHANNELS[2], self.CHANNELS[3], kernel_size=3, stride=2, padding=1, bias=True)
        layer4 = convbn(self.CHANNELS[3], self.CHANNELS[4], kernel_size=3, stride=2, padding=1, bias=True)
        pool = nn.AdaptiveAvgPool2d(self.POOL)
        self.layers = nn.Sequential(layer1, layer2, layer3, layer4, pool)
        self.nn = nn.Linear(self.POOL[0] * self.POOL[1] * self.CHANNELS[4], n_classes)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, return_feats=False):
        feats = self.layers(x).flatten(1)
        x = self.nn(self.dropout(feats))
        if return_feats:
            return x, feats
        return x        

doodle_model = CNN(1, 9)
real_model = CNN(3, 9)
real_val_set = RealDataset(train=False)
engine = Engine(real_val_set, doodle_model, real_model)

def doodle_search(img, topk):
    pretrained_paths = {
        "V1 MLP": ("", ""),
        "V2 CNN": ("", ""),
        "V3 CNN + Contrastive": ("", ""),
        "V4 CNN + Multi-contrastive": ("", ""),
        "V4 ConvNeXT + Multi-contrastive": ("", ""),
    }

    results = engine.query(img, topk=topk)

    output_transform = transforms.Compose([
        transforms.ToTensor()
    ]) 

    model = nn.Sequential(nn.Identity())
    output_tensor = torch.Tensor(topk, 3, 64, 64)
    for i, img in enumerate(results):
        output_tensor[i, :, :, :] = output_transform(img)

    save_image(output_tensor, "output.png", normalize=True)

    return 'output.png'

text = """
## Welcome to _Doogle_! 

Group 1 presents this reverse-image search engine that
takes in doodles and returns the best-matching real-life images to address 
your tip-of-the-tongue problem :D

**Note**: Please wait for database to be indexed when the app first loads up. Depending on the hardware you have, this might take a few minutes. Clearly, we are not able to beat Google's speeds :')

---

Work by: **Group 1**, Ai Bo, New Jun Jie, and Rishabh Anand
"""

demo = gr.Interface(
    fn=doodle_search,
    title="Doogle",
    article=text,
    inputs=[
        gr.inputs.Image(shape=(64, 64), image_mode="L", invert_colors=False, source="canvas"), 
        gr.inputs.Slider(minimum=5, maximum=20, step=1, label="Top K Best Matches")
    ],  
    outputs="image")

if __name__ == "__main__":
    demo.launch()