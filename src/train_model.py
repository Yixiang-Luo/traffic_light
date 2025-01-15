import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import torchvision.models as models


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return identity * out


class YOLOv8BackboneWithResNet50(nn.Module):
    def __init__(self):
        super(YOLOv8BackboneWithResNet50, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)

        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        self.coordatt = CoordAtt(2048)
        self.seblock = SEBlock(2048)

    def forward(self, x):
        x = self.resnet50(x)
        x = self.coordatt(x)
        x = self.seblock(x) 
        return x


class YOLOWithResNet50Backbone(YOLO):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super(YOLOWithResNet50Backbone, self).__init__(model=model, task=task, verbose=verbose)

        self.model.model[0] = YOLOv8BackboneWithResNet50()

    def forward(self, x):
        return super(YOLOWithResNet50Backbone, self).forward(x)


model = YOLOWithResNet50Backbone("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4
)
