# import onnx

# def print_model_structure(model_path):
#     # Load the ONNX model
#     model = onnx.load(model_path)

#     # Check the model
#     onnx.checker.check_model(model)

#     # Print the model graph
#     print("Model Graph:")
#     for i, node in enumerate(model.graph.node):
#         print(f"Node {i}:")
#         print(f"Name: {node.name}")
#         print(f"Op Type: {node.op_type}")
#         print("Inputs:")
#         for input_name in node.input:
#             print(f"    {input_name}")
#         print("Outputs:")
#         for output_name in node.output:
#             print(f"    {output_name}")
#         print()

# # Replace 'your_model.onnx' with the path to your ONNX model file
# model_path = './psy.onnx'
# print_model_structure(model_path)


import onnx


def get_value_info_shape(value_info):
    """Extract the shape from a ValueInfoProto"""
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def print_onnx_model_details(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    graph = model.graph
    print(f"Model graph name: {graph.name}")

    print("\nInputs:")
    for input in graph.input:
        shape = get_value_info_shape(input)
        print(f"Name: {input.name}, Shape: {shape}")

    print("\nOutputs:")
    for output in graph.output:
        shape = get_value_info_shape(output)
        print(f"Name: {output.name}, Shape: {shape}")

    print("\nNodes:")
    for node in graph.node:
        print(f"Name: {node.name}, OpType: {node.op_type}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")
        if node.op_type == "Conv":
            conv_params = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            print(f"Conv Params: {conv_params}")
        elif node.op_type == "BatchNormalization":
            bn_params = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            print(f"BatchNorm Params: {bn_params}")
        elif node.op_type == "Relu":
            print(f"ReLU activation")
        elif node.op_type == "Add":
            print(f"Add operation")
        # Add more operations as needed
        print()


# Replace 'your_model.onnx' with the path to your ONNX model file
model_path = 'psy.onnx'
print_onnx_model_details(model_path)

