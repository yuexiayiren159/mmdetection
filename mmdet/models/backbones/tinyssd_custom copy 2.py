# tinyssd_custom.py (融合后的版本)
# print(f"DEBUG: Importing/Executing tinyssd_custom.py - Full path: {__file__}")
# print(f"DEBUG: Timestamp: YOUR_NEW_UNIQUE_TIMESTAMP_HERE") # <<<--- 每次修改代码都更新这个标记！！！

from mmengine.model import BaseModule
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from typing import Tuple, List # 导入 List

# print("DEBUG: tinyssd_custom.py: Imports done, defining class...")

@MODELS.register_module()
class TinySSD_Custom(BaseModule):
    def __init__(self,
                 input_channels: int = 3,
                 # 定义每个阶段的配置: (output_channels, num_convs_in_block, is_feature_output)
                 # 最后一个元素如果是AdaptiveMaxPool2d，其num_convs可以设为0或特殊处理
                 stage_configs: List[Tuple[int, int, bool]] = [
                     (16, 2, False), # stage 0: out_c=16, 2 convs, 不是输出特征
                     (32, 2, False), # stage 1: out_c=32, 2 convs, 不是输出特征
                     (64, 2, True),  # stage 2: out_c=64, 2 convs, 是输出特征
                     (128, 2, True), # stage 3: out_c=128, 2 convs, 是输出特征
                     (128, 2, True), # stage 4: out_c=128, 2 convs, 是输出特征
                     (128, 2, True), # stage 5: out_c=128, 2 convs, 是输出特征
                     # 最后一个通常是全局池化，这里用一个特殊标记或不同的配置方式
                 ],
                 # 单独为最后一个全局池化层前的卷积层指定输出通道
                 final_conv_out_channels: int = 128, # 对应你原始代码中 AdaptiveMaxPool2d 前的128通道
                 init_cfg: dict = None): # init_cfg 应为 dict 或 None

        # print(f"DEBUG: TinySSD_Custom __init__ called. Timestamp: YOUR_NEW_UNIQUE_TIMESTAMP_HERE")
        super().__init__(init_cfg)

        self.stage_configs = stage_configs
        self.final_conv_out_channels = final_conv_out_channels
        self.blocks = nn.ModuleList()
        self._feature_block_indices = [] # 记录哪些 block 的索引对应输出特征

        self._make_network_layers(input_channels)

    def _make_downsample_block(self, in_c: int, out_c: int, num_convs: int = 2) -> nn.Sequential:
        layers = []
        current_channels = in_c
        for _ in range(num_convs):
            layers.append(nn.Conv2d(current_channels, out_c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_c
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _make_network_layers(self, current_c: int):
        """Helper function to build the network based on stage_configs."""
        for idx, config in enumerate(self.stage_configs):
            out_c, num_convs, is_feature = config
            self.blocks.append(self._make_downsample_block(current_c, out_c, num_convs))
            current_c = out_c # 更新当前通道数为该块的输出通道
            if is_feature:
                self._feature_block_indices.append(idx)

        # 添加最后一个 AdaptiveMaxPool2d 块
        # 它作用于最后一个 _make_downsample_block 的输出 (current_c)
        # 如果需要 AdaptiveMaxPool2d 前的卷积层有特定通道数，可以调整
        # 你原始代码是先经过若干 downsample block (输出128)，然后再 AdaptiveMaxPool2d
        # 这里的 current_c 就是最后一个 downsample block 的输出通道
        # 如果 final_conv_out_channels 与 current_c 不同，则需要一个额外的转换卷积
        adaptive_pool_input_channels = current_c
        if self.final_conv_out_channels != current_c:
            # 如果需要，在 AdaptiveMaxPool2d 之前添加一个或多个卷积来调整通道数
            # 为简化，这里假设 AdaptiveMaxPool2d 直接作用于最后一个 downsample_block 的输出
            # 或者，你需要一个不带 MaxPool2d 的卷积块来调整通道，然后再接 AdaptiveMaxPool2d
            # 这里的 final_conv_out_channels 应该等于最后一个 stage_configs 的 out_c
            # 如果你希望 AdaptiveMaxPool2d 的输入是特定的通道数，那么最后一个 stage_config 的 out_c 应该是那个值
            # 例如，如果最后一个 stage_config 的 out_c 是128，那么 adaptive_pool_input_channels 就是128
             pass # 假设最后一个 stage_config 的 out_c 已经是 final_conv_out_channels

        # 构建最后的 AdaptiveMaxPool2d 层，它本身作为一个 block
        # 它的输入通道是最后一个常规block的输出通道 (current_c)
        # 输出通道不变，但 H, W 变为 1, 1
        # 你原始代码是将 AdaptiveMaxPool2d 作为 self.blocks 的一个元素
        # 并且 idx=6 时，这个元素被选中作为输出
        # 所以我们直接将它追加到 blocks 列表
        final_pool_block = nn.AdaptiveMaxPool2d((1,1))
        self.blocks.append(final_pool_block)
        # 确保这个 block 也被标记为特征输出，如果它是你想要的最终特征之一
        # 对应你原始的 idx=6
        self._feature_block_indices.append(len(self.blocks) - 1)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # print(f"DEBUG: TinySSD_Custom forward called. Timestamp: YOUR_NEW_UNIQUE_TIMESTAMP_HERE")
        outputs = []
        current_f = x
        # 遍历所有 blocks (包括 downsample blocks 和最后的 AdaptiveMaxPool2d block)
        for i in range(len(self.blocks)):
            current_f = self.blocks[i](current_f)
            if i in self._feature_block_indices:
                outputs.append(current_f)
        return tuple(outputs)

# print(f"DEBUG: tinyssd_custom.py: Class TinySSD_Custom defined. Timestamp: YOUR_NEW_UNIQUE_TIMESTAMP_HERE")