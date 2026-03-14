import Base_Classes
import numpy as np

# layers:


# 线性层类
class Linear(Base_Classes.BaseLayer):
    # 初始化函数
    def __init__(self, input_dim: int, output_dim: int, bias: bool):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        w = np.random.normal(loc=0, scale=1, size=(input_dim, output_dim))
        self.params["w"] = Base_Classes.Parameter(w)
        # 根据输入决定是否需要偏置值
        if bias:
            b = np.random.normal(loc=0, scale=1, size=output_dim)
            self.params["b"] = Base_Classes.Parameter(b)
        else:
            pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # 清理上一次的缓存
        self.cache = {}
        self.cache["input"] = inputs
        output = inputs @ self.params["w"].value
        if self.bias:
            output = output + self.params["b"].value
        return output

    def backward(self, grad_of_output: np.ndarray) -> np.ndarray:
        x = self.cache["input"]
        # 1.计算损失对权重的梯度
        grad_w = x.T @ grad_of_output  # 形状(input_dim, output_dim)
        self.params["w"].grad = grad_w

        # 2. 计算损失对偏置的梯度
        if self.bias:
            grad_b = np.sum(
                grad_of_output,
                axis=0,
            )  # 形状(output_dim,)
            self.params["b"].grad = grad_b

        # 3. 计算损失对输入的梯度
        grad_input = (
            grad_of_output @ self.params["w"].value.T
        )  # 形状(batch_size, input_dim)
        return grad_input


# Dropout层类
class Dropout(Base_Classes.BaseLayer):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        # dropout层参数无需更新
        # 直接清空缓存即可
        self.cache = {}
        if self.training:
            random_tensor = np.random.random(inputs.shape)
            mask = (random_tensor > self.p).astype(np.float32)
            # 对保留的神经元进行缩放，保持训练和推理时期望值一致
            scale = 1.0 / (1.0 - self.p)
            # 储存掩码与缩放系数，用于反向传播
            self.cache["mask"] = mask
            self.cache["scale"] = scale
            return inputs * mask * scale

        else:
            return inputs

    def backward(self, grad_of_output: np.ndarray) -> np.ndarray:
        if self.training:
            mask = self.cache["mask"]
            scale = self.cache["scale"]
            return mask * scale * grad_of_output
        else:
            return grad_of_output


# ReLU类
class ReLU(Base_Classes.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        mask = (inputs > 0).astype(np.float32)
        self.cache["mask"] = mask
        return np.maximum(0, inputs)

    def backward(self, grad_of_output):
        mask = self.cache["mask"]
        grad_of_input = grad_of_output * mask
        return grad_of_input


# Sigmoid类
# 尚未考虑极端数值下的数值稳定性，留待后期完善
class Sigmoid(Base_Classes.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        output = 1 / (1 + np.exp(-inputs))
        self.cache["inputs"] = inputs
        self.cache["output"] = output
        return output

    def backward(self, grad_of_output):
        output = self.cache["output"]
        return output * (1 - output) * grad_of_output


# BatchNorm类 - 批归一化层
class Batchnorm(Base_Classes.BaseLayer):
    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ):
        """
        批归一化层
        参数:
            num_features: 输入特征的数量
            eps: 防止除零的小常数
            momentum: 用于计算 running_mean 和 running_var 的动量
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # 可学习参数: gamma(缩放) 和 beta(平移)
        gamma = np.ones(num_features)
        beta = np.zeros(num_features)
        self.params["gamma"] = Base_Classes.Parameter(gamma)
        self.params["beta"] = Base_Classes.Parameter(beta)
        # 运行时统计量 (不可学习参数)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        前向传播
        参数:
            inputs: 输入数据，形状为 (batch_size, num_features)
        返回:
            归一化后的输出
        """
        # 清空缓存
        self.cache = {}
        if self.training:
            # 训练模式
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            # 缓存中间变量用于反向传播
            self.cache["inputs"] = inputs
            self.cache["mean"] = mean
            self.cache["var"] = var
            self.cache["batch_size"] = inputs.shape[0]
            # 更新运行时统计量 (移动平均)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            # 推理模式
            # 使用运行时累积的统计量
            mean = self.running_mean
            var = self.running_var
        # 归一化: x_hat = (x - mean) / sqrt(var + eps)
        x_normalized = (inputs - mean) / np.sqrt(var + self.eps)
        # 缩放和平移: y = gamma * x_hat + beta
        output = (
            self.params["gamma"].value * x_normalized
            + self.params["beta"].value
        )
        # 缓存归一化后的值用于反向传播
        self.cache["x_normalized"] = x_normalized
        return output

    def backward(self, grad_of_output: np.ndarray) -> np.ndarray:
        """
        反向传播
        参数:
            grad_of_output: 损失对输出的梯度，形状为 (batch_size, num_features)
        返回:
            损失对输入的梯度
        """
        # 从缓存中获取中间变量
        x = self.cache["inputs"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_normalized = self.cache["x_normalized"]
        batch_size = self.cache["batch_size"]
        gamma = self.params["gamma"].value
        grad_gamma = np.sum(grad_of_output * x_normalized, axis=0)
        self.params["gamma"].grad = grad_gamma
        # dL/dbeta = sum(dL/dy)
        grad_beta = np.sum(grad_of_output, axis=0)
        self.params["beta"].grad = grad_beta
        # 中间变量
        std = np.sqrt(var + self.eps)
        x_centered = x - mean
        # dL/dx_hat = dL/dy * gamma
        grad_x_normalized = grad_of_output * gamma
        # dL/dvar 的梯度
        # dL/dvar = sum(dL/dx_hat * (x - mean) * (-0.5) * (var + eps)^(-3/2))
        grad_var = np.sum(
            grad_x_normalized
            * x_centered
            * (-0.5)
            * (var + self.eps) ** (-1.5),
            axis=0,
        )
        """
        dL/dmean 的梯度
        dL/dmean = sum(dL/dx_hat * (-1 / std))
        + dL/dvar * (-2 * mean(x) / batch_size)
        """
        grad_mean = np.sum(
            grad_x_normalized * (-1.0 / std), axis=0
        ) + grad_var * (-2.0 * np.mean(x_centered, axis=0))
        """
        dL/dx = dL/dx_hat * (1 / std) +
        dL/dvar * (2 * (x - mean) / batch_size) +
        dL/dmean * (1 / batch_size)
        """
        grad_input = (
            grad_x_normalized / std
            + grad_var * (2.0 * x_centered / batch_size)
            + grad_mean / batch_size
        )
        return grad_input


# loss_functions


class CrossEntropyLoss(Base_Classes.BaseLoss):
    def forward(self, prediction: np.ndarray, label: np.ndarray):
        # 将预测值转化为概率
        shifted_prediction = prediction - np.max(
            prediction, axis=-1, keepdims=True
        )
        exp_prediction = np.exp(shifted_prediction)
        sum = np.sum(exp_prediction, axis=-1, keepdims=True)
        probability = exp_prediction / sum
        self.cache["probability"] = probability
        self.cache["label"] = label
        batch_size = probability.shape[0]
        correct_class_probs = probability[np.arange(batch_size), label]
        loss = -np.log(correct_class_probs + 1e-12)
        loss = np.mean(loss)
        return loss

    def backward(self):
        probability = self.cache["probability"]
        label = self.cache["label"]
        batch_size = probability.shape[0]
        grad_of_out = probability.copy()
        grad_of_out[
            np.arange(batch_size), label
        ] -= 1  # probability - one_hot(label)即损失值对预测值的梯度
        grad_of_out = grad_of_out / batch_size
        return grad_of_out


# optimizers


# 随机梯度下降优化器
class SGD(Base_Classes.BaseOptimizer):
    def step(self):
        for item in self.params:
            item.value -= item.grad * self.lr


# Adam 优化器 (自适应矩估计)
class Adam(Base_Classes.BaseOptimizer):
    def __init__(
        self,
        lr: float,
        params: list,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Adam 优化器

        参数:
            lr: 学习率
            params: Parameter 对象列表
            betas: (beta1, beta2) 用于计算梯度的一阶和二阶矩估计的衰减率
            eps: 防止除零的小常数
        """
        super().__init__(lr=lr, params=params)
        self.beta1 = betas[0]  # 一阶矩估计的衰减率（通常为 0.9）
        self.beta2 = betas[1]  # 二阶矩估计的衰减率（通常为 0.999）
        self.eps = eps

        # 初始化矩估计
        self.m = {}  # 一阶矩估计（梯度的移动平均）
        self.v = {}  # 二阶矩估计（梯度平方的移动平均）
        self.t = 0  # 时间步

        for i, param in enumerate(self.params):
            self.m[i] = np.zeros_like(param.value)
            self.v[i] = np.zeros_like(param.value)

    def step(self):
        """
        更新参数

        Adam 算法:
        1. m = beta1 * m + (1 - beta1) * grad  (一阶矩估计)
        2. v = beta2 * v + (1 - beta2) * grad^2  (二阶矩估计)
        3. m_hat = m / (1 - beta1^t)  (偏差校正)
        4. v_hat = v / (1 - beta2^t)  (偏差校正)
        5. param = param - lr * m_hat / (sqrt(v_hat) + eps)
        """
        self.t += 1  # 增加时间步

        for i, param in enumerate(self.params):
            grad = param.grad

            # 更新一阶矩估计（梯度的指数移动平均）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 更新二阶矩估计（梯度平方的指数移动平均）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # 计算偏差校正后的估计
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # 更新参数
            param.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
