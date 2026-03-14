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
        grad_input = grad_of_output @ self.params["w"].value.T  # 形状(batch_size, input_dim)
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


# Batchnorm类
# 为保证进度,先不实现,留待后期优化时实现
class Batchnorm(Base_Classes.BaseLayer):
    pass


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
        grad_of_out[np.arange(batch_size), label] -= 1#probability - one_hot(label)即损失值对预测值的梯度
        grad_of_out = grad_of_out / batch_size
        return grad_of_out


# optimizers


# 随机梯度下降优化器
class SGD(Base_Classes.BaseOptimizer):
    def step(self):
        for item in self.params:
            item.value -= item.grad * self.lr
