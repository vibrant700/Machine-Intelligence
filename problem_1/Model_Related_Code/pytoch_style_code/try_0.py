"""
本代码为实现代码
经实验，本文件中代码与相关代码(Base_Classes.py和classes.py)部分(剩余部分尚未进行测试)可以跑通，损失可以稳定下降
也可作为参考代码
我会在注释中添加一些注意事项
——LXY
"""

import Base_Classes
import classes
import numpy as np
import support

# 定义学习率
lr = 0.01
# 定义学习轮数
epoch = 50
# 定义batch_size
batch_size = 64

# 准备数据
path = "working_dir\\Machine-Intelligence\\problem_1\\DATA\\MNIST"
train_features, train_label, test_features, test_labels = support.load_mnist(
    path
)
train_features = support.preprocess_images(train_features)
test_features = support.preprocess_images(test_features)

# 定义网络
# 注意层的传入需要传入一个层列表，而不是传入多个层
net = Base_Classes.Sequential(
    [
        classes.Linear(input_dim=784, output_dim=256, bias=True),
        classes.ReLU(),
        classes.Linear(input_dim=256, output_dim=128, bias=True),
        classes.ReLU(),
        classes.Linear(input_dim=128, output_dim=10, bias=True),
    ]
)
# 定义优化器
# 注意net.parameters()的括号
optimizer = classes.SGD(lr=lr, params=net.parameters())
# 定义损失函数
loss_function = classes.CrossEntropyLoss()

# 训练
num_samples = train_features.shape[0]
num_batches = num_samples // batch_size

for training_times in range(epoch):
    total_loss = 0
    indices = np.random.permutation(num_samples)
    train_features_shuffled = train_features[indices]
    train_labels_shuffled = train_label[indices]
    for i in range(num_batches):
        # 准备数据
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        features_batch = train_features_shuffled[start_idx:end_idx]
        labels_batch = train_labels_shuffled[start_idx:end_idx]
        # 生成预测值
        predict = net(features_batch)
        # 计算损失
        loss = loss_function(predict, labels_batch)
        # 对总损失进行累加(与训练无关，只是为了输出平均损失)
        total_loss += loss
        # 将所有参数的梯度置0
        optimizer.zero_grad()
        # 计算出损失关于输出值的梯度
        # 注意需要调用的是loss_function的backward方法，这点与pytorch不同
        grad = loss_function.backward()
        # 逐层反向传播
        # 需要将grad传入net的backward方法以使各参数梯度更新，这点也与pytorch不同
        net.backward(grad)
        # 对所有参数进行一次优化
        optimizer.step()
    average_loss = total_loss / num_batches
    print(f"完成第{training_times+1}轮训练，平均损失为{average_loss}")
