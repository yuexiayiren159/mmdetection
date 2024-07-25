import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out

class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        self.conv1 = ConvModule(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1
        self.stage1_0 = ResidualBlock(16, 32, stride=2, downsample=self._downsample(16, 32), groups=1)
        self.stage1_1 = ResidualBlock(32, 32, groups=1)

        # Stage 2
        self.stage2_0 = ResidualBlock(32, 64, stride=2, downsample=self._downsample(32, 64), groups=1)
        self.stage2_1 = ResidualBlock(64, 64, groups=1)

        # Stage 3
        self.stage3_0 = ResidualBlock(64, 96, stride=2, downsample=self._downsample(64, 96), groups=1)
        self.stage3_1 = ResidualBlock(96, 96, groups=1)
        self.stage3_2 = ResidualBlock(96, 96, groups=1)

        # Stage 4
        self.stage4_0 = ResidualBlock(96, 128, stride=2, downsample=self._downsample(96, 128), groups=1)
        self.stage4_1 = ResidualBlock(128, 128, groups=1)
        self.stage4_2 = ResidualBlock(128, 128, groups=1)
        self.stage4_3 = ResidualBlock(128, 128, groups=1)

        # Stage 5
        self.stage5_0 = ResidualBlock(128, 192, stride=2, downsample=self._downsample(128, 192), groups=1)
        self.stage5_1 = ResidualBlock(192, 192, groups=1)
        self.stage5_2 = ResidualBlock(192, 192, groups=1)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, 6)  # 修改全连接层的输入通道数

    def _downsample(self, in_channels, out_channels):
        return nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.stage1_0(x)
        x = self.stage1_1(x)

        x = self.stage2_0(x)
        x = self.stage2_1(x)

        x = self.stage3_0(x)
        x = self.stage3_1(x)
        x = self.stage3_2(x)

        x = self.stage4_0(x)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)

        x = self.stage5_0(x)
        x = self.stage5_1(x)
        x = self.stage5_2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 创建模型实例
model = CustomBackbone()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 96, 96)

# 导出为ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "custom_backbone.onnx", 
    input_names=["data_input"], 
    output_names=["output"],
    dynamic_axes={"data_input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
