# check_typing.py
import torch
from typing import Tuple # 必须导入

def my_func_new(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (x, x)

# def my_func_old(x: torch.Tensor) -> tuple[torch.Tensor, ...]: # 这是会出错的旧语法
#     return (x, x)

print("Typing check script running...")
if __name__ == '__main__':
    tensor_input = torch.randn(1,3,4,4)
    output_new = my_func_new(tensor_input)
    print(f"Output from new function type: {type(output_new)}, first element type: {type(output_new[0])}")
    print("New function definition seems OK if no error above this line.")

    # print("\nTesting old problematic syntax (expect TypeError if Python < 3.9 or if not using typing.Tuple for ellipsis):")
    # try:
    #     output_old = my_func_old(tensor_input) # 这行如果取消注释，在旧Python版本会报错
    # except TypeError as e:
    #     print(f"Caught expected TypeError for old syntax: {e}")