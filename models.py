import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, indim, classes):
        super().__init__()
        self.l1 = nn.Linear(indim, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        out = torch.softmax(self.l5(x), 1)

        return out

class CNN(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (3,3))
        self.conv2 = nn.Conv2d(32, 64, (3,3))
        self.mp1 = nn.MaxPool2d((2,2))
        self.mp2 = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(2304, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 128)
        self.l4 = nn.Linear(128, classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.mp1(self.conv1(x)))
        x = self.relu(self.mp1(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        out = torch.softmax(self.l4(x), 1)

        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x                

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, (7,7), padding=3, groups=dim)
        self.lin1 = nn.Linear(dim, 4 * dim)
        self.lin2 = nn.Linear(4 * dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        res_inp = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1) # NCHW -> NHWC
        x = self.ln(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.gelu(x)
        x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
        out = x + res_inp

        return out

class ConvNeXt(nn.Module):
    # TODO: ensure ConvNeXt is comparable to CNN (model size)
    # best to stick with one block
    def __init__(self, in_channels, classes, dims=[96, 192, 384, 768], depth=[]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, dims[0], (4,4), stride=4)
        self.blocks = nn.ModuleList()
        for dim in dims:
            stage = nn.Sequential(ConvNeXtBlock(dim))
            self.blocks.append(stage)
        self.project = nn.Linear(dims[-1], classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        out = self.project(x)
        return out

convnext = ConvNeXt(3, 10)
x = torch.rand(2, 3, 32, 32)
y = convnext(x)
print (y.shape)