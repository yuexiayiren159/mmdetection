import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.maxpool_layer_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )
        self.shortcut2_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(96)
        )
        self.relu = nn.ReLU(inplace=True)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, groups=96, stride=2, bias=False),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192)
        )

        self.layer6_1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(192)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192)
        )

        
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192)
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, groups=192, stride=2, bias=False),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384)
        )
        self.layer10_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(384)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=384, stride=1, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384)
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=384, stride=1, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384)
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=384, stride=1, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=192, bias=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=1, bias=False),
            nn.BatchNorm2d(384)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.maxpool_layer_1(x)

        x_shortcut1 = x
        x = self.layer2(x)
        x_shortcut1 = self.shortcut2_1(x_shortcut1)
        x = x + x_shortcut1
        x = self.relu(x)

        x_shortcut2 = x
        x = self.layer3(x)
        x = x + x_shortcut2 
        x = self.relu(x)

        x_shortcut3 = x
        x = self.layer4(x)
        x = x + x_shortcut3
        x = self.relu(x)

        x_shortcut4 = x
        x = self.layer5(x)
        x = x + x_shortcut4
        x = self.relu(x)

        x_branch_1 = x   #
        x = self.layer6(x) + self.layer6_1(x_branch_1)
        x = self.relu(x)

        x_shortcut5 = x
        x = self.layer7(x) + x_shortcut5
        x = self.relu(x)

        x_shortcut6 = x
        x = self.layer8(x) + x_shortcut6
        x = self.relu(x)

        x_branch2 = x    # 
        x = self.layer9(x) + x_branch2
        x = self.relu(x)

        x_shortcut8 = x
        x = self.layer10(x) + self.layer10_1(x_shortcut8)
        x = self.relu(x)

        x_shortcut9 = x
        x = self.layer11(x) + x_shortcut9
        x = self.relu(x)

        x_shortcut10 = x
        x = self.layer12(x) + x_shortcut10
        x = self.relu(x)

        x_shortcut11 = x
        x = self.layer13(x) + x_shortcut11
        x = self.relu(x)

        return x

# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel().to(device)
# input = torch.randn(1,3,416,736)
# output = model(input)
# print("output.shape: ",output.shape)

from torchsummary import summary

summary(model, (3, 416, 736), device=str(device))


