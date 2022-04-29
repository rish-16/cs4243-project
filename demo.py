import torch
import gradio as gr
from engine import Engine
from model_training import *

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
        layer1_2 = convbn(self.CHANNELS[1], self.CHANNELS[1], kernel_size=3, stride=1, padding=0, bias=True)
        layer2_2 = convbn(self.CHANNELS[2], self.CHANNELS[2], kernel_size=3, stride=1, padding=0, bias=True)
        layer3_2 = convbn(self.CHANNELS[3], self.CHANNELS[3], kernel_size=3, stride=1, padding=0, bias=True)
        layer4_2 = convbn(self.CHANNELS[4], self.CHANNELS[4], kernel_size=3, stride=1, padding=0, bias=True)
        self.layers = nn.Sequential(layer1, layer2, layer3, layer4, pool)
        self.nn = nn.Linear(self.POOL[0] * self.POOL[1] * self.CHANNELS[4], n_classes)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, return_feats=False):
        feats = self.layers(x).flatten(1)
        x = self.nn(self.dropout(feats))
        if return_feats:
            return x, feats
        return x

def doodle_search(img, model_choice, topk):
    pretrained_paths = {
        "V1 MLP": ("", ""),
        "V2 CNN": ("", ""),
        "V3 CNN + Contrastive": ("", ""),
        "V4 CNN + Multi-contrastive": ("", ""),
        "V4 ConvNeXT + Multi-contrastive": ("", ""),
    }

    models = {
        "V1 MLP": (None, None),
        "V2 CNN": (None, None),
        "V3 CNN + Contrastive": (None, None),
        "V4 CNN + Multi-contrastive": (None, None),
        "V4 ConvNeXT + Multi-contrastive": (None, None),
    }

    doodle_path, real_path = pretrained_paths[model_choice]
    doodle_model_class, real_model_class = models[model_choice]
    # doodle_model = load_model(doodle_model_class, doodle_path)
    # real_model = load_model(real_model_class, real_path)

    doodle_model = CNN(1, 9)
    real_model = CNN(3, 9)

    real_val_set = RealDataset(train=False)
    engine = Engine(real_val_set, doodle_model, real_model)
    results = engine.query(img, topk=topk)

    return results

demo = gr.Interface(
    fn=doodle_search,
    inputs=[
        gr.inputs.Image(shape=(64, 64), image_mode="L", invert_colors=False, source="canvas"), 
        gr.inputs.Dropdown(
            ["V1 MLP", "V2 CNN", "V3 CNN + Contrastive", "V4 CNN + Multi-contrastive", "V5 ConvNeXT + Multi-contrastive"], 
            label="Choose your model"
        ),
        gr.inputs.Slider(minimum=5, maximum=20, step=1, label="Top K Best Matches")
    ],  
    outputs="image")

if __name__ == "__main__":
    demo.launch()