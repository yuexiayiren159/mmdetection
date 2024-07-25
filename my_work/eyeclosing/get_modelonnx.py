import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.scale1 = nn.Parameter(torch.ones(16))
        self.bias1 = nn.Parameter(torch.zeros(16))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.scale2 = nn.Parameter(torch.ones(32))
        self.bias2 = nn.Parameter(torch.zeros(32))

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.scale3 = nn.Parameter(torch.ones(32))
        self.bias3 = nn.Parameter(torch.zeros(32))

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.scale4 = nn.Parameter(torch.ones(64))
        self.bias4 = nn.Parameter(torch.zeros(64))

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, self.scale1, self.bias1, training=False)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.batch_norm(x, self.bn2.running_mean, self.bn2.running_var, self.scale2, self.bias2, training=False)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.batch_norm(x, self.bn3.running_mean, self.bn3.running_var, self.scale3, self.bias3, training=False)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.batch_norm(x, self.bn4.running_mean, self.bn4.running_var, self.scale4, self.bias4, training=False)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x


# 创建模型实例
model = CustomModel()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 48, 48)

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "custom_backbone.onnx",
    input_names=["data"],
    output_names=["softmax"],
    dynamic_axes={"data": {0: "batch_size", 2: "height", 3: "width"}, "softmax": {0: "batch_size"}},
    opset_version=11,
    training=2
)
