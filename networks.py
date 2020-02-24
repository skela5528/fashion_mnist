import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
        super(ConvBlock, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x, verbose=False):
        out = self.block(x)
        if verbose:
            print(x.shape, "->", out.shape)
        return out


class Classifier(nn.Module):
    """Fully connected classifier module."""
    def __init__(self, in_features, middle_features=64, out_features=10, n_hidden_layers=1):
        super(Classifier, self).__init__()

        layers = list()
        is_last_layer = not bool(n_hidden_layers)
        layers.append(nn.Linear(in_features=in_features,
                                out_features=out_features if is_last_layer else middle_features))

        while n_hidden_layers > 0:
            is_last_layer = n_hidden_layers <= 1
            layers.append(nn.Linear(in_features=middle_features,
                                    out_features=out_features if is_last_layer else middle_features))
            n_hidden_layers -= 1
        self.fc = nn.Sequential(*layers)

    def forward(self, x, verbose=False):
        out = self.fc(x)
        if verbose:
            print(x.shape, "->", out.shape)
        return out


class Net3Conv(nn.Module):
    def __init__(self):
        super(Net3Conv, self).__init__()
        backbone_layers = list()
        backbone_layers.append(ConvBlock(in_channels=1, out_channels=32))
        backbone_layers.append(nn.MaxPool2d(kernel_size=2))
        backbone_layers.append(ConvBlock(in_channels=32, out_channels=16))
        backbone_layers.append(nn.MaxPool2d(kernel_size=2))
        backbone_layers.append(ConvBlock(in_channels=16, out_channels=32))
        backbone_layers.append(nn.MaxPool2d(kernel_size=2))
        backbone_layers.append(nn.ReLU())

        self.backbone = nn.Sequential(*backbone_layers)
        self.classifier = Classifier(in_features=32 * 3 * 3)

    def forward(self, x):
        x = self.backbone(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        out = self.classifier(x)
        return out


class Net9Conv(nn.Module):
    def __init__(self):
        super(Net9Conv, self).__init__()
        backbone_layers = list()
        backbone_layers.append(ConvBlock(in_channels=1, out_channels=16))
        backbone_layers.append(ConvBlock(in_channels=16, out_channels=32))
        backbone_layers.append(ConvBlock(in_channels=32, out_channels=64))
        backbone_layers.append(nn.MaxPool2d(kernel_size=2))
        backbone_layers.append(ConvBlock(in_channels=64, out_channels=64))
        backbone_layers.append(ConvBlock(in_channels=64, out_channels=32))
        backbone_layers.append(ConvBlock(in_channels=32, out_channels=64))
        backbone_layers.append(nn.MaxPool2d(kernel_size=2))
        backbone_layers.append(ConvBlock(in_channels=64, out_channels=32))
        backbone_layers.append(ConvBlock(in_channels=32, out_channels=32))
        backbone_layers.append(ConvBlock(in_channels=32, out_channels=32))
        backbone_layers.append(nn.MaxPool2d(kernel_size=2))
        backbone_layers.append(nn.ReLU())

        self.backbone = nn.Sequential(*backbone_layers)
        self.classifier = Classifier(in_features=32 * 3 * 3)

    def forward(self, x):
        x = self.backbone(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        out = self.classifier(x)
        return out


def get_efficientnet_pretrained_on_imagenet(model_name="efficientnet-b0", num_classes=10, in_channels=1):
    """For more details: https://github.com/lukemelas/EfficientNet-PyTorch/"""
    model = EfficientNet.from_pretrained("efficientnet-b2", num_classes=10, in_channels=1)
    # model.set_swish(memory_efficient=False)
    return model.cuda()
