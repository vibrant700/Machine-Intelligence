import numpy as np

"""
初始化的作用是加速收敛,避免梯度消失或爆炸
Xavier:适合sigmoid、tanh、softmax等激活函数
Kaiming:适合ReLU及其衍生的激活函数
"""
# initialization 初始化函数


# Xavier均匀分布初始化
def XavierUniform(shape: tuple) -> np.ndarray:
    dim_in = shape[0]
    dim_out = shape[1]
    range = np.sqrt(6 / (dim_in + dim_out))
    result = np.random.uniform(low=-range, high=range, size=shape)
    return result


# Xavier正态分布初始化
def XavierNormal(shape: tuple) -> np.ndarray:
    shape = input.shape
    dim_in = shape[0]
    dim_out = shape[1]
    range = np.sqrt(2 / (dim_in + dim_out))
    result = np.random.normal(loc=0, scale=range, size=shape)
    return result


# Kaiming均匀分布初始化
def KaimingUniform(shape: tuple) -> np.ndarray:
    shape = input.shape
    dim_in = shape[0]
    range = np.sqrt(6 / dim_in)
    result = np.random.uniform(low=-range, high=range, size=shape)
    return result


# Kaiming正态分布初始化
def KaimingNormal(shape: tuple) -> np.ndarray:
    shape = input.shape
    dim_in = shape[0]
    range = np.sqrt(2 / dim_in)
    result = np.random.normal(loc=0, scale=range, size=shape)
    return result
