import pickle

import support
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# 指定设备为gpu
device = torch.device("cuda")
# 指定学习率
lr = 5e-4
# 指定训练轮数
epoch = 200
# 指定数据集路径
path = "working_dir\\Machine-Intelligence\\problem_1\\DATA\\MNIST"
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
    print(f"""第{i+1}轮训练后
平均损失为{average_loss}
在验证集上的正确率为{accuracy*100}%
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
print(f"训练完成,在测试集上的正确率为{accuracy*100}%")
model_path = "working_dir\\Machine-Intelligence\\problem_1\\mnist_service\\backend\\CNN.pkl"
with open(model_path, "wb") as f:
    pickle.dump(net, f)
print(f"模型文件成功保存至:{model_path}")
