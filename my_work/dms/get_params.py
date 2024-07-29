import onnx

def calculate_params(onnx_model_path):
    model = onnx.load(onnx_model_path)
    total_params = 0

    for initializer in model.graph.initializer:
        param_count = 1
        for dim in initializer.dims:
            param_count *= dim
        total_params += param_count

    return total_params / 1e6  # 转换为百万（M）

onnx_model_path = 'yolov3_xceptionb0_bsd_sigmastar.onnx'
params = calculate_params(onnx_model_path)
print(f"Params (M): {params:.2f}")
