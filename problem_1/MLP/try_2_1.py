"""
完成第1轮训练，平均损失为0.38715653092707003
完成第2轮训练，平均损失为0.16216959160075337
完成第3轮训练，平均损失为0.12036656541816997
完成第4轮训练，平均损失为0.09467135321336216
完成第5轮训练，平均损失为0.0752021101281959
完成第6轮训练，平均损失为0.06704950736249347
完成第7轮训练，平均损失为0.057556078158610455
完成第8轮训练，平均损失为0.04840801112868924
完成第9轮训练，平均损失为0.046038156532543596
完成第10轮训练，平均损失为0.04070942572477834
完成第11轮训练，平均损失为0.03450417988559963
完成第12轮训练，平均损失为0.03346139696424205
完成第13轮训练，平均损失为0.02996395176996723
完成第14轮训练，平均损失为0.026899338729137117
完成第15轮训练，平均损失为0.02587068872422731
完成第16轮训练，平均损失为0.023805462358858654
完成第17轮训练，平均损失为0.02120175010112484
完成第18轮训练，平均损失为0.021244605406676845
完成第19轮训练，平均损失为0.019719592505007943
完成第20轮训练，平均损失为0.018557915320648082
0       974     17      6       9003    0.9828  0.9939  0.9883
1       1126    11      9       8854    0.9903  0.9921  0.9912
2       1011    19      21      8949    0.9816  0.9797  0.9806
3       994     12      16      8978    0.9881  0.9842  0.9861
4       957     11      25      9007    0.9886  0.9745  0.9815
5       878     15      14      9093    0.9832  0.9843  0.9838
6       945     12      13      9030    0.9875  0.9864  0.9869
7       1013    25      15      8947    0.9759  0.9854  0.9806
8       953     24      21      9002    0.9754  0.9784  0.9769
9       982     21      27      8970    0.9791  0.9732  0.9761

宏平均 (Macro-Average):
  精确率: 0.9833 (98.33%)
  召回率: 0.9832 (98.32%)
  F1分数: 0.9832

微平均 (Micro-Average):
  精确率: 0.9833 (98.33%)
  召回率: 0.9833 (98.33%)
  F1分数: 0.9833

总体准确率: 0.9833 (98.33%)

分析说明：
  宏平均和微平均接近，说明各类别样本量相对均衡
"""

import Base_Classes
import classes
import numpy as np
import support
from torch.utils.tensorboard import SummaryWriter

# 定义学习率
lr = 0.001  # Adam 通常使用较小的学习率
# 定义学习轮数
epoch = 20
# 定义batch_size
batch_size = 128


# 准备数据
path = "working_dir\\Machine-Intelligence\\problem_1\\DATA\\MNIST"
train_features, train_label, test_features, test_labels = support.load_mnist(
    path
)
train_features = support.preprocess_images(train_features)
test_features = support.preprocess_images(test_features)

# 定义网络
net = Base_Classes.Sequential(
    [
        classes.Linear(
            input_dim=784,
            output_dim=512,
            bias=True,
            weight_init="kaiming_normal",
        ),
        classes.Batchnorm(512),
        classes.Dropout(0.2),
        classes.ReLU(),
        classes.Linear(
            input_dim=512,
            output_dim=256,
            bias=True,
            weight_init="kaiming_normal",
        ),
        classes.Batchnorm(256),
        classes.Dropout(0.3),
        classes.ReLU(),
        classes.Linear(
            input_dim=256,
            output_dim=128,
            bias=True,
            weight_init="kaiming_normal",
        ),
        classes.Batchnorm(128),
        classes.Dropout(0.4),
        classes.ReLU(),
        classes.Linear(
            input_dim=128,
            output_dim=10,
            bias=True,
            weight_init="kaiming_normal",
        ),
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
    # 设置为训练模式
    net.train()
    for i in range(num_batches):
        # 准备数据
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        features_batch = train_features[start_idx:end_idx]
        labels_batch = train_label[start_idx:end_idx]
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
        print("w完成一个batch")
    average_loss = total_loss / num_batches
    print(f"完成第{training_times+1}轮训练，平均损失为{average_loss}")
    writer.add_scalar("average_loss", average_loss, training_times)
    # 设置为预测模式
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
