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

        # self.layer29_33 = nn.Sequential(
        #     nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1,padding=1, bias=False),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1,padding=1, bias=False),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True)
        # )

        # self.layer34_35 = nn.Sequential(
        #     nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=2, stride=2, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True)
        # )

        # self.layer36_37 = nn.Sequential(
        #     nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=384, out_channels=30, kernel_size=1, stride=1, bias=False),
        # )

        # self.layer38 = nn.Sequential(
        #     nn.Conv2d(in_channels=288, out_channels=96, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=192, out_channels=96, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True)
        # )

        # self.layer39 = nn.Sequential(
        #     nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True)
        # )
        # self.out2 = nn.Sequential(
        #     nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=192, out_channels=30, kernel_size=1, stride=1, bias=False),
        # )

        # self.layer40 = nn.Sequential(
        #     nn.Conv2d(in_channels=144, out_channels=48, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(96),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(96),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=96, out_channels=30, kernel_size=1, stride=1, bias=False),
        # )

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
        # print("x.shape: ", x.shape)
        # print("x_shortcut2.shape: ", x_shortcut2.shape)

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
        # print("x_branch_1.shape: ", x_branch_1.shape)

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

        # x = self.layer29_33(x)
        # x = self.relu(x)

        # x_out1 = x
        # x = self.layer34(x)
        # x = self.layer35(x)
        # x_out1 = self.layer36(x_out1)
        # out1 = self.layer37(x_out1)
        # x = torch.concat((x,x_branch2), dim=1)
        # print("out1.shape: ", out1.shape)

        # x = self.layer38(x)

        # x_out2 = x
        # x = self.layer39(x)
        # out2 = self.out2(x_out2)

        print("x.shape: ", x.shape)

        # print("out2.shape: ", out2.shape)

        # x = torch.concat((x,x_branch_1), dim=1)

        # out3 = self.layer40(x)
        # print("out3.shape: ", out3.shape)

        return x

# 实例化模型
model = CustomModel()
input = torch.randn(1,3,416,736)
# output1, output2, output3 = model(input)
# print("output1.shape: ",output1.shape)
# print("output2.shape: ",output2.shape)
# print("output3.shape: ",output3.shape)

output = model(input)
# print("output.shape: ",output.shape)