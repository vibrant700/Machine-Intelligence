# -*- coding: utf-8 -*-
"""
基于 NumPy 实现的 BP 神经网络（PyTorch 风格）
用于 MNIST 手写数字识别
"""

import numpy as np
import os
import struct


# ============================================
# 第一部分：数据加载和预处理
# ============================================

def load_mnist_images(filename):
    """
    加载 MNIST 图像文件

    参数:
        filename: 图像文件路径

    返回:
        images: 图像数组，形状为 (数量, 28, 28)
    """
    with open(filename, 'rb') as f:
        # 读取文件头信息
        _ = struct.unpack('>I', f.read(4))[0]  # 魔数，不使用
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    return images


def load_mnist_labels(filename):
    """
    加载 MNIST 标签文件

    参数:
        filename: 标签文件路径

    返回:
        labels: 标签数组，形状为 (数量,)
    """
    with open(filename, 'rb') as f:
        # 读取文件头信息
        _ = struct.unpack('>I', f.read(4))[0]  # 魔数，不使用
        _ = struct.unpack('>I', f.read(4))[0]  # 标签数量，不使用

        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def load_mnist(path):
    """
    加载完整的 MNIST 数据集

    参数:
        path: MNIST 数据集所在目录

    返回:
        train_images: 训练图像
        train_labels: 训练标签
        test_images: 测试图像
        test_labels: 测试标签
    """
    print("正在加载训练数据...")
    train_images = load_mnist_images(os.path.join(path, 'train-images.idx3-ubyte'))
    train_labels = load_mnist_labels(os.path.join(path, 'train-labels.idx1-ubyte'))

    print("正在加载测试数据...")
    test_images = load_mnist_images(os.path.join(path, 't10k-images.idx3-ubyte'))
    test_labels = load_mnist_labels(os.path.join(path, 't10k-labels.idx1-ubyte'))

    return train_images, train_labels, test_images, test_labels


def preprocess_images(images):
    """
    预处理图像数据

    步骤:
    1. 将 28x28 的图像展平成 784 维向量
    2. 将像素值从 0-255 归一化到 0-1

    参数:
        images: 原始图像数组，形状为 (数量, 28, 28)

    返回:
        processed_images: 处理后的图像，形状为 (数量, 784)
    """
    num_samples = images.shape[0]

    # 将图像展平：从 (N, 28, 28) 变成 (N, 784)
    flattened = images.reshape(num_samples, -1)

    # 归一化：将 0-255 的像素值缩放到 0-1 之间
    normalized = flattened.astype(np.float32) / 255.0

    return normalized


# ============================================
# 第二部分：神经网络层（PyTorch 风格）
# ============================================

class Module:
    """
    所有层的基类

    每个层都需要继承这个类并实现 forward 和 backward 方法
    """

    def __init__(self):
        # 存储参数（权重和偏置）
        self.params = {}
        # 存储梯度
        self.grads = {}
        # 存储输入（用于反向传播）
        self.input_cache = {}

    def forward(self, x):
        """前向传播"""
        raise NotImplementedError("子类必须实现 forward 方法")

    def backward(self, dout):
        """反向传播"""
        raise NotImplementedError("子类必须实现 backward 方法")

    def __call__(self, x):
        """让对象可以像函数一样调用"""
        return self.forward(x)


class Linear(Module):
    """
    全连接层（线性层）

    实现：y = x @ W + b
    这是神经网络中最基础的层
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        # 使用 Xavier 初始化
        # 这种初始化方法有利于梯度传播
        scale = np.sqrt(2.0 / input_size)
        self.params['W'] = np.random.randn(input_size, output_size) * scale
        self.params['b'] = np.zeros(output_size)

        # 初始化梯度（形状与参数相同）
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])

    def forward(self, x):
        """
        前向传播

        计算：output = input @ weights + bias

        参数:
            x: 输入数据，形状为 (batch_size, input_size)

        返回:
            output: 输出数据，形状为 (batch_size, output_size)
        """
        # 缓存输入，反向传播时需要用到
        self.input_cache['x'] = x

        # 计算：x @ W + b
        output = np.dot(x, self.params['W']) + self.params['b']

        return output

    def backward(self, dout):
        """
        反向传播

        计算对输入和参数的梯度

        参数:
            dout: 从后面传过来的梯度

        返回:
            dx: 对输入的梯度
        """
        # 从缓存中获取输入
        x = self.input_cache['x']
        batch_size = x.shape[0]

        # 计算梯度
        # dL/dW = x^T @ dout
        self.grads['W'] = np.dot(x.T, dout) / batch_size

        # dL/db = sum(dout, axis=0)
        self.grads['b'] = np.sum(dout, axis=0) / batch_size

        # dL/dx = dout @ W^T
        dx = np.dot(dout, self.params['W'].T)

        return dx


class Sigmoid(Module):
    """
    Sigmoid 激活函数层

    公式：f(x) = 1 / (1 + e^(-x))
    输出范围：0 到 1
    """

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入

        返回:
            sigmoid(x)
        """
        # 防止数值溢出
        x = np.clip(x, -500, 500)

        # 计算 sigmoid
        output = 1.0 / (1.0 + np.exp(-x))

        # 缓存输出，反向传播时需要用到
        self.input_cache['output'] = output

        return output

    def backward(self, dout):
        """
        反向传播

        sigmoid 的导数：f'(x) = f(x) * (1 - f(x))

        参数:
            dout: 从后面传过来的梯度

        返回:
            dx: 对输入的梯度
        """
        # 从缓存中获取输出
        output = self.input_cache['output']

        # 计算梯度
        dx = dout * output * (1.0 - output)

        return dx


class ReLU(Module):
    """
    ReLU 激活函数层

    公式：f(x) = max(0, x)
    现代神经网络最常用的激活函数
    """

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入

        返回:
            max(0, x)
        """
        # 缓存输入大于0的位置
        self.input_cache['mask'] = (x > 0)

        # ReLU：大于0的保留，小于0的变为0
        output = x * self.input_cache['mask']

        return output

    def backward(self, dout):
        """
        反向传播

        ReLU 的导数：x>0 时为 1，x<=0 时为 0

        参数:
            dout: 从后面传过来的梯度

        返回:
            dx: 对输入的梯度
        """
        # 只在输入大于0的位置传递梯度
        dx = dout * self.input_cache['mask']

        return dx


class SoftmaxCrossEntropyLoss:
    """
    Softmax 交叉熵损失函数

    将 softmax 和交叉熵合并在一起，数值更稳定
    这是多分类问题最常用的损失函数
    """

    def __init__(self):
        # 缓存中间结果
        self.cache = {}

    def forward(self, logits, labels):
        """
        前向传播：计算损失

        参数:
            logits: 网络输出（未经过 softmax），形状为 (batch_size, num_classes)
            labels: 真实标签（整数），形状为 (batch_size,)

        返回:
            loss: 平均损失值
        """
        batch_size = logits.shape[0]

        # 数值稳定的 softmax
        # 减去最大值防止 exp() 溢出
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)

        # 计算 softmax
        exp_scores = np.exp(logits_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 计算交叉熵损失
        # -log(probs[真实类别])
        correct_log_probs = -np.log(probs[range(batch_size), labels] + 1e-10)
        loss = np.sum(correct_log_probs) / batch_size

        # 缓存概率和标签，反向传播时需要
        self.cache['probs'] = probs
        self.cache['labels'] = labels
        self.cache['batch_size'] = batch_size

        return loss

    def backward(self):
        """
        反向传播：计算梯度

        返回:
            dout: 对 logits 的梯度
        """
        probs = self.cache['probs']
        labels = self.cache['labels']
        batch_size = self.cache['batch_size']

        # 创建 one-hot 编码
        dout = probs.copy()
        dout[range(batch_size), labels] -= 1.0
        dout /= batch_size

        return dout


# ============================================
# 第三部分：神经网络模型
# ============================================

class NeuralNetwork(Module):
    """
    神经网络模型

    采用类似 PyTorch 的设计，可以像搭积木一样构建网络
    """

    def __init__(self):
        super().__init__()
        # 存储所有的层
        self.layers = []

    def add(self, layer):
        """
        添加一个层

        参数:
            layer: 要添加的层
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        前向传播

        按顺序通过所有层

        参数:
            x: 输入数据

        返回:
            output: 网络输出
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, dout):
        """
        反向传播

        按相反顺序通过所有层

        参数:
            dout: 输出的梯度

        返回:
            dx: 输入的梯度
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def zero_grad(self):
        """
        清零所有层的梯度

        在每次反向传播前需要调用
        """
        for layer in self.layers:
            if hasattr(layer, 'grads'):
                for key in layer.grads:
                    layer.grads[key] = np.zeros_like(layer.grads[key])

    def update_params(self, learning_rate):
        """
        使用梯度下降更新所有参数

        参数:
            learning_rate: 学习率
        """
        for layer in self.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for key in layer.params:
                    # 梯度下降：参数 = 参数 - 学习率 * 梯度
                    layer.params[key] -= learning_rate * layer.grads[key]

    def predict(self, x):
        """
        预测

        参数:
            x: 输入数据

        返回:
            predictions: 预测的类别
        """
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def compute_accuracy(self, x, labels):
        """
        计算准确率

        参数:
            x: 输入数据
            labels: 真实标签

        返回:
            accuracy: 准确率（0-1之间）
        """
        predictions = self.predict(x)
        accuracy = np.mean(predictions == labels)
        return accuracy


# ============================================
# 第四部分：训练和评估
# ============================================

def train_epoch(model, x_train, y_train, criterion, learning_rate, batch_size=64):
    """
    训练一个 epoch

    参数:
        model: 神经网络模型
        x_train: 训练数据
        y_train: 训练标签
        criterion: 损失函数
        learning_rate: 学习率
        batch_size: 批次大小

    返回:
        average_loss: 平均损失
    """
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
    total_loss = 0.0

    # 打乱数据
    indices = np.random.permutation(num_samples)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    # 小批次训练
    for batch_idx in range(num_batches):
        # 获取当前批次
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        x_batch = x_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]

        # 前向传播
        logits = model.forward(x_batch)
        loss = criterion.forward(logits, y_batch)
        total_loss += loss * len(x_batch)

        # 反向传播
        model.zero_grad()
        dout = criterion.backward()
        model.backward(dout)

        # 更新参数
        model.update_params(learning_rate)

    # 返回平均损失
    average_loss = total_loss / num_samples
    return average_loss


def train(model, x_train, y_train, x_test, y_test,
          epochs=50, batch_size=64, learning_rate=0.1):
    """
    训练神经网络

    参数:
        model: 神经网络模型
        x_train: 训练数据
        y_train: 训练标签
        x_test: 测试数据
        y_test: 测试标签
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 创建损失函数
    criterion = SoftmaxCrossEntropyLoss()

    print("\n开始训练...")
    print(f"训练样本数: {x_train.shape[0]}")
    print(f"测试样本数: {x_test.shape[0]}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    print("=" * 80)

    for epoch in range(epochs):
        # 训练一个 epoch
        train_loss = train_epoch(model, x_train, y_train, criterion, learning_rate, batch_size)

        # 每 5 轮打印一次结果
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 计算准确率
            train_acc = model.compute_accuracy(x_train, y_train)
            test_acc = model.compute_accuracy(x_test, y_test)

            print(f"轮次 {epoch + 1}/{epochs}:")
            print(f"  训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc*100:.2f}%")
            print(f"  测试集 - 准确率: {test_acc*100:.2f}%")

    print("=" * 80)
    print("训练完成！")


def evaluate_per_class(y_true, y_pred, num_classes=10):
    """
    评估每个类别的识别效果

    计算每个数字的精确率、召回率、F1分数

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
    """
    print("\n" + "=" * 80)
    print("每个数字的识别结果：")
    print("=" * 80)
    print("数字    TP      FP      FN      TN      精确率  召回率  F1分数")
    print("-" * 70)

    # 存储每个类别的指标
    precisions = []
    recalls = []
    f1_scores = []

    for digit in range(num_classes):
        # 计算混淆矩阵的四个值
        tp = np.sum((y_true == digit) & (y_pred == digit))  # 真阳性
        fp = np.sum((y_true != digit) & (y_pred == digit))  # 假阳性
        fn = np.sum((y_true == digit) & (y_pred != digit))  # 假阴性
        tn = np.sum((y_true != digit) & (y_pred != digit))  # 真阴性

        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"{digit}       {tp:<6}  {fp:<6}  {fn:<6}  {tn:<6}  {precision:.4f}  {recall:.4f}  {f1:.4f}")

    return precisions, recalls, f1_scores


def compute_overall_metrics(y_true, y_pred):
    """
    计算整体识别指标

    包括宏平均、微平均、总体准确率

    参数:
        y_true: 真实标签
        y_pred: 预测标签
    """
    num_classes = 10

    # 获取每个类别的指标
    precisions, recalls, f1_scores = evaluate_per_class(y_true, y_pred, num_classes)

    # ========== 计算宏平均 ==========
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    # ========== 计算微平均 ==========
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for digit in range(num_classes):
        total_tp += np.sum((y_true == digit) & (y_pred == digit))
        total_fp += np.sum((y_true != digit) & (y_pred == digit))
        total_fn += np.sum((y_true == digit) & (y_pred != digit))

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # ========== 计算总体准确率 ==========
    accuracy = np.mean(y_true == y_pred)

    # 打印结果
    print("\n" + "=" * 80)
    print("整体识别结果：")
    print("=" * 80)

    print(f"\n宏平均 (Macro-Average):")
    print(f"  精确率: {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"  召回率: {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"  F1分数: {macro_f1:.4f}")

    print(f"\n微平均 (Micro-Average):")
    print(f"  精确率: {micro_precision:.4f} ({micro_precision*100:.2f}%)")
    print(f"  召回率: {micro_recall:.4f} ({micro_recall*100:.2f}%)")
    print(f"  F1分数: {micro_f1:.4f}")

    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 80)

    # 分析宏平均和微平均的差异
    print("\n分析说明：")
    if abs(macro_f1 - micro_f1) < 0.01:
        print("  宏平均和微平均接近，说明各类别样本量相对均衡")
    elif macro_f1 < micro_f1:
        print("  宏平均低于微平均，建议检查样本量较少的类别")
    else:
        print("  微平均低于宏平均，建议检查样本量较多的类别")


# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    print("BP 神经网络 - MNIST 手写数字识别")

    # 数据路径
    data_path = "problem_1/DATA/MNIST"

    # 第一步：加载数据
    print("\n第一步：加载 MNIST 数据集")
    train_images, train_labels, test_images, test_labels = load_mnist(data_path)
    print(f"训练集: {train_images.shape[0]} 张图像")
    print(f"测试集: {test_images.shape[0]} 张图像")

    # 第二步：预处理数据
    print("\n第二步：预处理数据")
    x_train = preprocess_images(train_images)
    x_test = preprocess_images(test_images)
    print(f"图像处理: {train_images.shape[1]}x{train_images.shape[2]} -> {x_train.shape[1]} 维向量")
    print(f"像素归一化: 0-255 -> 0-1")
    print(f"标签格式: 整数标签 (0-9)")

    # 第三步：创建神经网络
    print("\n第三步：创建神经网络")
    model = NeuralNetwork()

    # 像搭积木一样构建网络
    model.add(Linear(784, 128))   # 输入层 -> 隐含层1 (784 -> 128)
    model.add(Sigmoid())            # 激活函数
    model.add(Linear(128, 10))    # 隐含层2 -> 输出层 (128 -> 10)

  

    # 第四步：训练网络
    print("\n第四步：训练网络")
    train(
        model, x_train, train_labels,
        x_test, test_labels,
        epochs=50,
        batch_size=64,
        learning_rate=0.1
    )

    # 第五步：预测和评估
    print("\n第五步：预测和评估")
    print("正在对测试集进行预测...")
    y_pred = model.predict(x_test)

    # 评估结果
    compute_overall_metrics(test_labels, y_pred)

    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
