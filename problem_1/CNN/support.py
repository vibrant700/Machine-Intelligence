import os
import struct

import numpy as np


# 用于图像预处理
def load_mnist_images(filename):
    """
    加载 MNIST 图像文件

    参数:
        filename: 图像文件路径

    返回:
        images: 图像数组，形状为 (数量, 28, 28)
    """
    with open(filename, "rb") as f:
        # 读取文件头信息
        _ = struct.unpack(">I", f.read(4))[0]  # 魔数，不使用
        num_images = struct.unpack(">I", f.read(4))[0]
        num_rows = struct.unpack(">I", f.read(4))[0]
        num_cols = struct.unpack(">I", f.read(4))[0]

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
    with open(filename, "rb") as f:
        # 读取文件头信息
        _ = struct.unpack(">I", f.read(4))[0]  # 魔数，不使用
        _ = struct.unpack(">I", f.read(4))[0]  # 标签数量，不使用

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
    train_images = load_mnist_images(
        os.path.join(path, "train-images.idx3-ubyte")
    )
    train_labels = load_mnist_labels(
        os.path.join(path, "train-labels.idx1-ubyte")
    )

    test_images = load_mnist_images(
        os.path.join(path, "t10k-images.idx3-ubyte")
    )
    test_labels = load_mnist_labels(
        os.path.join(path, "t10k-labels.idx1-ubyte")
    )

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
