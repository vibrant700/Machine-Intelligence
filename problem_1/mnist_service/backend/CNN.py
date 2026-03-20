import support
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

"""
20轮
0       973     15      7       9005    0.9848  0.9929  0.9888
1       1132    19      3       8846    0.9835  0.9974  0.9904
2       1028    8       4       8960    0.9923  0.9961  0.9942
3       1004    10      6       8980    0.9901  0.9941  0.9921
4       978     26      4       8992    0.9741  0.9959  0.9849
5       885     12      7       9096    0.9866  0.9922  0.9894
6       945     3       13      9039    0.9968  0.9864  0.9916
7       1014    9       14      8963    0.9912  0.9864  0.9888
8       965     6       9       9020    0.9938  0.9908  0.9923
9       964     4       45      8987    0.9959  0.9554  0.9752

宏平均 (Macro-Average):
  精确率: 0.9889 (98.89%)
  召回率: 0.9887 (98.87%)
  F1分数: 0.9888

微平均 (Micro-Average):
  精确率: 0.9888 (98.88%)
  召回率: 0.9888 (98.88%)
  F1分数: 0.9888

总体准确率: 0.9888 (98.88%)

分析说明：
  宏平均和微平均接近，说明各类别样本量相对均衡
"""

"""
200轮
0       978     7       2       9013    0.9929  0.9980  0.9954
1       1134    6       1       8859    0.9947  0.9991  0.9969
2       1028    4       4       8964    0.9961  0.9961  0.9961
3       1006    8       4       8982    0.9921  0.9960  0.9941
4       977     8       5       9010    0.9919  0.9949  0.9934
5       885     7       7       9101    0.9922  0.9922  0.9922
6       949     2       9       9040    0.9979  0.9906  0.9942
7       1022    5       6       8967    0.9951  0.9942  0.9946
8       971     2       3       9024    0.9979  0.9969  0.9974
9       995     6       14      8985    0.9940  0.9861  0.9900

宏平均 (Macro-Average):
  精确率: 0.9945 (99.45%)
  召回率: 0.9944 (99.44%)
  F1分数: 0.9944

微平均 (Micro-Average):
  精确率: 0.9945 (99.45%)
  召回率: 0.9945 (99.45%)
  F1分数: 0.9945

总体准确率: 0.9945 (99.45%)

分析说明：
  宏平均和微平均接近，说明各类别样本量相对均衡
"""

# 自动检测设备（有GPU用GPU，没有用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 指定学习率
lr = 5e-4
# 指定训练轮数
epoch = 200
# 指定数据集路径
path = "..\\..\\DATA\\MNIST"
# 读取数据集
train_val_features, train_val_labels, test_features, test_labels = (
    support.load_mnist(path)
)
# 转换为张量
train_val_features = torch.tensor(train_val_features, dtype=torch.float32)
train_val_labels = torch.tensor(train_val_labels, dtype=torch.long)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)
# 转换为(batch_size,1,28,28)
train_val_features = train_val_features.reshape(-1, 1, 28, 28)
test_features = test_features.reshape(-1, 1, 28, 28)
# 创建dataset
train_val_dataset = TensorDataset(train_val_features, train_val_labels)
# 划分出验证集
len_of_train_features = int(0.9 * len(train_val_features))
len_of_val_features = len(train_val_features) - len_of_train_features
train_dataset, val_dataset = random_split(
    train_val_dataset, [len_of_train_features, len_of_val_features]
)
# 定义dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# 定义网络
net = nn.Sequential(
    nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        padding=1,
        stride=1,
    ),  # 大小不变
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 变为14*14
    nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=1,
    ),  # 大小不变
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 大小变为7*7
    nn.Conv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=1,
    ),  # 大小不变
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
net = net.to(device)

# 定义损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
# 定义优化器
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",  # 特征越大越好
    factor=0.5,  # 每次衰减的系数
    patience=10,  # 若10轮内，参数没有突破原有最大/最小值，学习率就会衰减
    cooldown=5,  # 衰减后5轮内不会再次检测
    threshold=0.0001,  # 只有超过0.0001的变化才会被视为改善
    min_lr=1e-7,  # 学习率不小于1e-7
)
for i in range(epoch):
    net.train()
    total_loss = 0
    total_times = 0
    for features, lables in train_dataloader:
        features = features.to(device)
        lables = lables.to(device)
        output = net(features)
        loss = loss_function(output, lables)
        total_loss += loss.item()
        total_times += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / total_times
    net.eval()
    correct = 0
    total = 0
    for features, lables in val_dataloader:
        features = features.to(device)
        lables = lables.to(device)
        output = net(features)
        _, predicted = torch.max(output.data, 1)
        total += lables.size(0)
        correct += (predicted == lables).sum().item()
    accuracy = correct / total
    scheduler.step(accuracy)
    current_lr = scheduler.get_last_lr()[0]
    print(f"""第{i + 1}轮训练后
平均损失为{average_loss}
在验证集上的正确率为{accuracy * 100}%
当前学习率: {current_lr}""")
# 在测试集上进行测试
net.eval()
with torch.no_grad():
    final_output = net(test_features.to(device))
    _, predict = torch.max(final_output.data, 1)
predict = predict.cpu()
correct = (predict == test_labels).sum().item()
total = len(test_features)
accuracy = correct / total

support.compute_overall_metrics(test_labels.numpy(), predict.numpy())
print(f"训练完成,在测试集上的正确率为{accuracy * 100}%")
model_path = "CNN.pkl"
torch.save(net, model_path)
print(f"模型文件成功保存至:{model_path}")
