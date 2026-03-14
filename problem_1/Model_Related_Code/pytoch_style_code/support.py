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


# 用于评估性能
def evaluate_per_class(y_true, y_pred, num_classes=10):
    """
    评估每个类别的识别效果
    计算每个数字的精确率、召回率、F1分数
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数
    """
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
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        # print(f"{digit} {tp:<6} {fp:<6} {fn:<6} {tn:<6} ")
        # print(f"{precision:.4f} {recall:.4f} {f1:.4f}")
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
    precisions, recalls, f1_scores = evaluate_per_class(
        y_true, y_pred, num_classes
    )
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for digit in range(num_classes):
        total_tp += np.sum((y_true == digit) & (y_pred == digit))
        total_fp += np.sum((y_true != digit) & (y_pred == digit))
        total_fn += np.sum((y_true == digit) & (y_pred != digit))
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )
    accuracy = np.mean(y_true == y_pred)
    # 打印结果
    print("\n宏平均 (Macro-Average):")
    print(f"  精确率: {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"  召回率: {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    print(f"  F1分数: {macro_f1:.4f}")

    print("\n微平均 (Micro-Average):")
    print(f"  精确率: {micro_precision:.4f} ({micro_precision*100:.2f}%)")
    print(f"  召回率: {micro_recall:.4f} ({micro_recall*100:.2f}%)")
    print(f"  F1分数: {micro_f1:.4f}")

    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    # 分析宏平均和微平均的差异
    print("\n分析说明：")
    if abs(macro_f1 - micro_f1) < 0.01:
        print("  宏平均和微平均接近，说明各类别样本量相对均衡")
    elif macro_f1 < micro_f1:
        print("  宏平均低于微平均，建议检查样本量较少的类别")
    else:
        print("  微平均低于宏平均，建议检查样本量较多的类别")
