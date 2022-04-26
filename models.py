import torch.nn as nn


class ExampleMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super(ExampleMLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, hid_dim)
        self.l4 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, return_feats=False):
        x = x.flatten(1)  # flatten a pic into a vector
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.l3(x)
        feat = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)

        if return_feats:
            return x, feat

        return x


def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ExampleCNN(nn.Module):
    CHANNELS = [64, 128, 192, 256, 512]
    POOL = (1, 1)

    def __init__(self, in_c, num_classes, dropout=0.2, add_layers=False):
        super().__init__()
        layer1 = convbn(in_c, self.CHANNELS[1], kernel_size=3, stride=2, padding=1, bias=True)
        layer2 = convbn(self.CHANNELS[1], self.CHANNELS[2], kernel_size=3, stride=2, padding=1, bias=True)
        layer3 = convbn(self.CHANNELS[2], self.CHANNELS[3], kernel_size=3, stride=2, padding=1, bias=True)
        layer4 = convbn(self.CHANNELS[3], self.CHANNELS[4], kernel_size=3, stride=2, padding=1, bias=True)
        pool = nn.AdaptiveAvgPool2d(self.POOL)
        self.layers = nn.Sequential(layer1, layer2, layer3, layer4, pool)

        if add_layers:
            layer1_2 = convbn(self.CHANNELS[1], self.CHANNELS[1], kernel_size=3, stride=1, padding=1, bias=True)
            layer2_2 = convbn(self.CHANNELS[2], self.CHANNELS[2], kernel_size=3, stride=1, padding=1, bias=True)
            layer3_2 = convbn(self.CHANNELS[3], self.CHANNELS[3], kernel_size=3, stride=1, padding=1, bias=True)
            layer4_2 = convbn(self.CHANNELS[4], self.CHANNELS[4], kernel_size=3, stride=1, padding=1, bias=True)
            self.layers = nn.Sequential(layer1, layer1_2, layer2, layer2_2, layer3, layer3_2, layer4, layer4_2, pool)

        self.nn = nn.Linear(self.POOL[0] * self.POOL[1] * self.CHANNELS[4], num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, return_feats=False):
        feats = self.layers(x).flatten(1)
        x = self.nn(self.dropout(feats))

        if return_feats:
            return x, feats

        return x


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
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.mp1 = nn.MaxPool2d((2, 2))
        self.mp2 = nn.MaxPool2d((2, 2))
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
        self.normalized_shape = (normalized_shape,)

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
        self.conv1 = nn.Conv2d(dim, dim, (7, 7), padding=3, groups=dim)
        self.lin1 = nn.Linear(dim, 4 * dim)
        self.lin2 = nn.Linear(4 * dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        res_inp = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.ln(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.gelu(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        out = x + res_inp

        return out


class ConvNeXt(nn.Module):
    def __init__(self, in_channels, classes, block_dims=[192, 384]):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, block_dims[0], kernel_size=2, stride=2),
            ConvNeXtBlock(block_dims[0]),
            nn.Conv2d(block_dims[0], block_dims[1], kernel_size=2, stride=2),
            ConvNeXtBlock(block_dims[1]),
        )
        self.block_dims = block_dims
        self.project = nn.Linear(block_dims[-1], classes)

    def forward(self, x, return_feats=False):
        feats = self.blocks(x)
        x = feats.view(-1, self.block_dims[-1], 16 * 16).mean(2)
        out = self.project(x)

        return out, feats if return_feats else out
