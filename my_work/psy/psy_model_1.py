import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)

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
        # print("residual.shape: ",residual.shape)
        # print("out.shape: ",out.shape)
        out += residual
        out = F.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, next_channels, stride=1, groups=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, next_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(next_channels)

    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # print("convblock_out.shape: ",out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        # print("convblock_out.shape: ",out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        # print("residual.shape: ",residual.shape)
        # print("convblock_out.shape: ",out.shape)
        # out += residual
        out = F.relu(out)
        return out

class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1
        self.stage1_0 = ResidualBlock(16, 32, stride=1, groups=32)
        self.stage2_0 = ConvBlock(16, 64,next_channels=24, stride=2, groups=64)

        # Stage 2
        self.stage2_1 = ResidualBlock(24, 96, groups=96)
        self.stage3_0 = ConvBlock(24, 96,next_channels=32, stride=2, groups=96)
        # Stage 3
        self.stage3_1 = ResidualBlock(32, 128, groups=128)
        self.stage3_2 = ResidualBlock(32, 128, groups=128)
        self.stage3_3 = ConvBlock(32, 128,next_channels=64, stride=2, groups=128)
        # Stage 4
        self.stage4_0 = ResidualBlock(64, 256, stride=1, groups=128)
        self.stage4_1 = ResidualBlock(64, 256, groups=256)
        self.stage4_2 = ResidualBlock(64, 256, groups=256)

        self.stage4_3 = ConvBlock(64, 256,next_channels=96, groups=256)

        # Stage 5
        self.stage5_0 = ResidualBlock(96, 192, stride=1, groups=192)
        self.stage5_1 = ResidualBlock(96, 192, groups=192)
        # self.stage5_2 = ResidualBlock(96, 192, groups=192)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.stage1_0(x)
        x = self.stage2_0(x)

        x = self.stage2_1(x)
        x = self.stage3_0(x)

        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)

        x = self.stage4_0(x)
        x = self.stage4_1(x)
        x = self.stage4_2(x)

        x = self.stage4_3(x)

        x = self.stage5_0(x)
        x = self.stage5_1(x)
        # x = self.stage5_2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)
    
    def _initialize_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)
        init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.fc.bias, 0)

# 创建模型实例
model = CustomBackbone()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 96, 96)

output = model(dummy_input)
print(output.shape)

# 导出为ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "custom_backbone_2.onnx", 
    input_names=["data_input"], 
    output_names=["sigmoid"],
    dynamic_axes={
        "data_input": {0: "1", 2: "96", 3: "96"},
        "sigmoid": {0: "1", 1: "6"}},
    opset_version=11,
    training=2
)

