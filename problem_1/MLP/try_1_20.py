'''
指标	数值
总轮数	20 轮
初始损失	0.204672
最终损失	0.021043
最终准确率	97.87%

lr	0.01
epoch	20 epochs
batch	64	每批处理 64 个样本
激活函数	ReLU

Linear(784, 128)
Batchnorm(num_features=128)
ReLU()
Linear(128, 64)
Batchnorm(num_features=64)
ReLU()
Linear(64, 10)
'''

import Base_Classes
import classes
import numpy as np
import support
from torch.utils.tensorboard import SummaryWriter

# 定义学习率
lr = 0.01  # Adam 通常使用较小的学习率
# 定义学习轮数
epoch = 20
# 定义batch_size
batch_size = 64


# 准备数据
path = "d:\\machine-intelligence\\problem_1\\DATA\\MNIST"
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
        classes.Linear(784, 128, bias=True),
        classes.Batchnorm(num_features=128),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活

 

        # 输出层: 128 -> 10
        classes.Linear(128, 64, bias=True),
        classes.Batchnorm(num_features=64),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活
        classes.Linear(64, 10, bias=True),
    ]
)
# 定义优化器
optimizer = classes.Adam(lr=lr, params=net.parameters())
# 定义损失函数
loss_function = classes.CrossEntropyLoss()
# 定义总结写入器
writer = SummaryWriter("log")

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
    writer.add_scalar("average_loss", average_loss, training_times)
    # 设置为预测模式
    if training_times % 5 == 0:
        net.eval()
        prediction = net(test_features)
        final_prediction = np.argmax(prediction, axis=1)
        correct = np.sum(final_prediction == test_labels)
        total_test_num = len(test_labels)
        accuracy = correct / total_test_num
        # 显示并记录数据
        writer.add_scalar("accuracy", accuracy, training_times)

# 评估
prediction = net(test_features)
final_prediction = np.argmax(prediction, axis=1)
support.compute_overall_metrics(test_labels, final_prediction)
