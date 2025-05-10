# test_backbone.py
import torch
from mmengine.config import Config
from mmdet.registry import MODELS
# from mmengine.runner import Runner # 移除，因为在独立测试 backbone 时非必需

# 确保自定义模块能被导入
# 假设 tinyssd_custom.py 位于 MMDetection 的标准路径或 PYTHONPATH 中
# 例如：e:\workspace\lanyun_work\openmmlab\mmdetection\mmdet\models\backbones\tinyssd_custom.py
# 并且其 __init__.py 中有 from .tinyssd_custom import TinySSD_Custom

def main():
    # --- 方法1: 直接实例化 (不通过MMDetection的Config系统，用于快速单元测试) ---
    print("--- Testing Backbone Directly ---")
    try:
        # 这个配置应该与你之前能够成功运行并打印出正确特征图的那个
        # TinySSD_Custom 的 __init__ 定义相匹配。
        # 特别是参数名 `out_channels`
        backbone_cfg_direct = dict(
            type='TinySSD_Custom',
            input_channels=3,
            # 这个参数名和值需要与你 tinyssd_custom.py 中 TinySSD_Custom 的 __init__ 匹配
            # 并且这个列表的长度和值，代表了期望的5个输出特征图的通道数
            out_channels=[64, 128, 128, 128, 128], # 与之前测试成功的版本一致
            init_cfg=None # 或者你的初始化配置
        )

        # 确保 MODELS 注册表已填充。通常导入 mmdet.models 会触发。
        try:
            import mmdet.models # noqa
        except ImportError:
            print("Warning: Failed to import mmdet.models. Make sure MMDetection is correctly installed and discoverable.")


        backbone = MODELS.build(backbone_cfg_direct)
        print("Backbone instantiated successfully (Directly).")

        dummy_input = torch.randn(2, 3, 256, 256)
        print(f"Dummy input shape: {dummy_input.shape}")

        # 如果你的 backbone 内部有 print 语句 (如 DEBUG: Timestamp:)
        # 在这里调用 forward 之前，确保它们已经被正确设置
        output_features = backbone(dummy_input)

        print("\nOutput features (Directly):")
        if isinstance(output_features, tuple):
            print(f"Number of output feature maps: {len(output_features)}")
            for i, feat in enumerate(output_features):
                print(f"  Feature map {i} shape: {feat.shape}")
        else:
            print(f"Output is not a tuple! Shape: {output_features.shape}")
            print("Error: Backbone must return a tuple of feature maps.")

        # 检查通道数是否与 backbone_cfg_direct['out_channels'] 匹配
        # 因为 backbone 的 __init__ 使用了 out_channels 参数来定义期望的输出
        expected_channels_list = backbone_cfg_direct['out_channels']
        if isinstance(output_features, tuple) and len(output_features) == len(expected_channels_list):
            all_channels_match = True
            for i, feat in enumerate(output_features):
                if feat.shape[1] != expected_channels_list[i]:
                    print(f"Channel mismatch for feature {i}: Expected {expected_channels_list[i]}, Got {feat.shape[1]}")
                    all_channels_match = False
            if all_channels_match:
                print("All output feature map channels match expected 'out_channels' from direct config.")
        elif not isinstance(output_features, tuple):
             print(f"Output feature count mismatch: Expected {len(expected_channels_list)} feature maps in a tuple, but got a single tensor or other type.")
        else: # Is a tuple, but length doesn't match
            print(f"Output feature count mismatch: Expected {len(expected_channels_list)}, Got {len(output_features)} feature maps.")


    except Exception as e:
        print(f"Error during direct backbone test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50 + "\n")

    # --- 方法2: 通过MMDetection的Config系统加载 (更完整) ---
    print("--- Testing Backbone via MMDetection Config ---")
    # 请确保这里的路径是正确的，指向你的主配置文件 ssd_custom.py
    # 该配置文件中的 model.backbone 部分应该与上面的 backbone_cfg_direct 对应
    config_file = 'E:/workspace/lanyun_work/openmmlab/mmdetection/configs/ssd/ssd_custom.py'

    try:
        cfg = Config.fromfile(config_file)

        if cfg.get('custom_imports', None):
            import mmengine.utils
            mmengine.utils.import_modules_from_strings(**cfg['custom_imports'])
        else:
            try:
                import mmdet.models # noqa
            except ImportError:
                print("Warning: Failed to import mmdet.models during config test. MMDetection might not be fully discoverable.")


        if 'backbone' in cfg.model and 'type' in cfg.model.backbone:
            # 确保 cfg.model.backbone 中的参数与 backbone_cfg_direct 一致
            # 特别是 'out_channels' (如果你的 TinySSD_Custom 的 __init__ 用的是这个参数名)
            print(f"Configuring backbone from cfg.model.backbone: {cfg.model.backbone}")
            backbone_from_cfg = MODELS.build(cfg.model.backbone)
            print("Backbone instantiated successfully (via Config).")

            dummy_input_cfg = torch.randn(2, 3, 256, 256)
            print(f"Dummy input shape (for config test): {dummy_input_cfg.shape}")

            output_features_cfg = backbone_from_cfg(dummy_input_cfg)

            print("\nOutput features (via Config):")
            if isinstance(output_features_cfg, tuple):
                print(f"Number of output feature maps: {len(output_features_cfg)}")
                for i, feat in enumerate(output_features_cfg):
                    print(f"  Feature map {i} shape: {feat.shape}")
                    if 'bbox_head' in cfg.model and 'in_channels' in cfg.model.bbox_head:
                        expected_head_in_channels = cfg.model.bbox_head.in_channels
                        if i < len(expected_head_in_channels):
                            if feat.shape[1] == expected_head_in_channels[i]:
                                print(f"    Channel {feat.shape[1]} matches head.in_channels[{i}]")
                            else:
                                print(f"    Channel MISMATCH! Feat channel: {feat.shape[1]}, Head.in_channels[{i}]: {expected_head_in_channels[i]}")
                        else:
                            print(f"    More features output by backbone ({len(output_features_cfg)}) than expected by head.in_channels ({len(expected_head_in_channels)}).")
                    else:
                        print("    bbox_head.in_channels not found in config for comparison.")
            else:
                print(f"Output is not a tuple! Shape: {output_features_cfg.shape}")
                print("Error: Backbone must return a tuple of feature maps.")
        else:
            print("Error: cfg.model.backbone or its 'type' not found in config.")


    except Exception as e:
        print(f"Error during config-based backbone test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()