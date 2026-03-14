"""
本代码为实现代码
经实验，本文件中代码与相关代码(Base_Classes.py和classes.py)部分(剩余部分尚未进行测试)可以跑通，损失可以稳定下降
也可作为参考代码
我会在注释中添加一些注意事项
——LXY

优化器说明:
- SGD (随机梯度下降): 简单稳定，但需要仔细调参
  * 学习率推荐: 0.01
  * 优点: 代码简单，泛化性能好
  * 缺点: 收敛较慢，对学习率敏感

- Adam (自适应矩估计): 自动调整学习率，收敛快
  * 学习率推荐: 0.001 (比 SGD 小)
  * 优点: 收敛快，对参数不敏感，适合大多数场景
  * 缺点: 代码稍复杂，泛化性能可能略差于 SGD

针对本代码结果的说明:
总体准确率: 0.9829 (98.29%)
从17轮训练开始平均损失落入0.01-0.001量级
随后总体下降，局部表现出随机游走
本模型在未使用Dropout层的情况下,训练轮数过多
但是没有发生过拟合现象，具体原因未知
可以考虑引入tensorboard模块以更好地观察每轮的损失和正确率
"""

import Base_Classes
import classes
import numpy as np
import support

# 定义学习率
lr = 0.001  # Adam 通常使用较小的学习率
# 定义学习轮数
epoch = 10
# 定义batch_size
batch_size = 64


# 准备数据
path = "problem_1\\DATA\\MNIST"
train_features, train_label, test_features, test_labels = support.load_mnist(
    path
)
train_features = support.preprocess_images(train_features)
test_features = support.preprocess_images(test_features)

# 定义网络
# 层的顺序: Linear -> BatchNorm -> ReLU
# BatchNorm 对未激活的值进行归一化
net = Base_Classes.Sequential(
    [
        # 第一层: 784 -> 256
        classes.Linear(input_dim=784, output_dim=256, bias=True),
        classes.Batchnorm(num_features=256),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活
        # 第二层: 256 -> 128
        classes.Linear(input_dim=256, output_dim=128, bias=True),
        classes.Batchnorm(num_features=128),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活
        # 输出层: 128 -> 10
        classes.Linear(input_dim=128, output_dim=10, bias=True),
    ]
)
# 定义优化器
optimizer = classes.Adam(lr=lr, params=net.parameters())
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
    # 设置为训练模式
    net.train()
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
# 评估
prediction = net(test_features)
final_prediction = np.argmax(prediction, axis=1)
support.compute_overall_metrics(test_labels, final_prediction)
