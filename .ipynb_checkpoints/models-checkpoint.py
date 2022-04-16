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
        self.ln = LayerNorm(dim)
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
    def __init__(self, in_chans=3, num_classes=10, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(1.0)
        self.head.bias.data.mul_(1.0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = ConvNeXt()
    x = torch.rand(1, 3, 64, 64)
    y = model(x)
    print (y.shape)