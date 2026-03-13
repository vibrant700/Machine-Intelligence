from abc import ABC, abstractmethod

import numpy as np


class Parameter:
    def __init__(
        self,
        value: np.ndarray,
        name: str = None,
    ):
        """
        初始化参数
        data: 参数数据
        name: 参数名称
        """
        self.value = value
        self.grad = np.zeros_like(value)
        self.name = name

    def zero_grad(self):
        """清空梯度"""
        self.grad.fill(0)


class BaseLayer(ABC):
    def __init__(self):
        self.params = {}  # 用于储存Parameter对象
        self.cache = {}  # 用于储存前向传播的中间结果
        self.training = True

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        实现正向传播
        同时缓存反向传播需要的中间变量
        """
        pass

    @abstractmethod
    def backward(self, grad_of_output: np.ndarray) -> np.ndarray:
        """
        实现反向传播
        输入:损失函数对本层输出的梯度
        返回:损失函数对本层输入的梯度
        同时保存梯度
        """
        pass

    def parameters(self):
        """
        返回本层的所有可变参数
        """
        return self.params

    def train(self):
        """设置训练模式"""
        self.training = True

    def eval(self):
        """设置评估模式"""
        self.training = False


class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add_layer(self, layer):
        """
        用于添加层
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        实现前向传播,同时储存每一层的输出
        由于Layer类的forward函数中已经缓存了中间结果，因此本处不缓存
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """
        实现反向传播
        返回值为损失值对整个网络输入的梯度
        """
        grad = grad_output
        #
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params_list = []
        for layer in self.layers:
            layer_params = layer.parameters()  # 字典
            for param in layer_params.values():  # 字典的值
                params_list.append(param)
        return params_list

    def __call__(self, x):
        return self.forward(x)


class BaseLoss(ABC):
    def __init__(self):
        self.cache = {}

    @abstractmethod
    def forward(self, prediction, label):
        """
        前向传播,计算当前的损失值
        """
        pass

    @abstractmethod
    def backward(self):
        """
        反向传播,计算损失值对预测值的梯度
        """
        pass

    def __call__(self, prediction, label):
        """使损失函数可以像函数一样调用"""
        return self.forward(prediction, label)


class BaseOptimizer(ABC):
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def step(self, params: list[Parameter]):
        """
        根据梯度更新参数
        params: Parameter对象的列表
        """
        pass

    def zero_grad(self, params: list[Parameter]):
        """
        用于将梯度置0
        """
        for item in params:
            item.zero_grad()
