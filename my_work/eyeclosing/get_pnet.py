import torch.nn as nn
import torch

class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.custom_block1 = CustomBlock(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)


        self.custom_block2 = CustomBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)


        self.custom_block3 = CustomBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)


        self.custom_block4 = CustomBlock(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)

        self.pool_5 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)

        self.InnerProduct = nn.Linear(in_features=128, out_features=2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.custom_block1(x)
        x = self.pool_1(x)

        x = self.custom_block2(x)
        x = self.pool_2(x)

        x = self.custom_block3(x)
        x = self.pool_3(x)

        x = self.custom_block4(x)
        x = self.pool_4(x)

        x = self.conv5(x)
        x = self.pool_5(x)

        x = x.view(x.size(0), -1)
        x = self.InnerProduct(x)
        x = self.softmax(x)

        return x

model = PNet()
dummy_input = torch.randn(1, 3, 48, 48)
output = model(dummy_input)
print(output.shape)

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "pnet.onnx",
    input_names=["datas"],
    output_names=["softmax"],
    dynamic_axes={"data": {0: "batch_size", 2: "height", 3: "width"}, "softmax": {0: "batch_size"}},
    opset_version=10,
    training=2
)