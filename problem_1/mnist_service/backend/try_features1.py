"""
完成第1轮训练，平均损失为0.15583822287326562
第1次训练后，在测试集上,正确率为0.9808
完成第2轮训练，平均损失为0.0764240998613753
第2次训练后，在测试集上,正确率为0.9801
完成第3轮训练，平均损失为0.058754484867988295
第3次训练后，在测试集上,正确率为0.985
完成第4轮训练，平均损失为0.043502442190670315
第4次训练后，在测试集上,正确率为0.9856
完成第5轮训练，平均损失为0.03931087251992354
第5次训练后，在测试集上,正确率为0.9858
完成第6轮训练，平均损失为0.032475918727426265
第6次训练后，在测试集上,正确率为0.9841
完成第7轮训练，平均损失为0.030722694352722187
第7次训练后，在测试集上,正确率为0.9867
完成第8轮训练，平均损失为0.027885315395727348
第8次训练后，在测试集上,正确率为0.9853
完成第9轮训练，平均损失为0.025452929014202475
第9次训练后，在测试集上,正确率为0.985
完成第10轮训练，平均损失为0.02139830004759935
第10次训练后，在测试集上,正确率为0.9862
完成第11轮训练，平均损失为0.020781230145849986
第11次训练后，在测试集上,正确率为0.9839
完成第12轮训练，平均损失为0.018877623355060758
第12次训练后，在测试集上,正确率为0.9881
完成第13轮训练，平均损失为0.01513638632346458
第13次训练后，在测试集上,正确率为0.9861
完成第14轮训练，平均损失为0.015356584010015169
第14次训练后，在测试集上,正确率为0.9871
完成第15轮训练，平均损失为0.013608276124025818
第15次训练后，在测试集上,正确率为0.9873
完成第16轮训练，平均损失为0.012811346581995812
第16次训练后，在测试集上,正确率为0.9874
完成第17轮训练，平均损失为0.01164356388040784
第17次训练后，在测试集上,正确率为0.9874
完成第18轮训练，平均损失为0.010754479578172963
第18次训练后，在测试集上,正确率为0.9882
完成第19轮训练，平均损失为0.010072992196566717
第19次训练后，在测试集上,正确率为0.987
完成第20轮训练，平均损失为0.01107625929193475
第20次训练后，在测试集上,正确率为0.9892
0       971     6       9       9014    0.9939  0.9908  0.9923
1       1130    13      5       8852    0.9886  0.9956  0.9921
2       1020    5       12      8963    0.9951  0.9884  0.9917
3       1006    19      4       8971    0.9815  0.9960  0.9887
4       963     3       19      9015    0.9969  0.9807  0.9887
5       881     5       11      9103    0.9944  0.9877  0.9910
6       948     20      10      9022    0.9793  0.9896  0.9844
7       1006    6       22      8966    0.9941  0.9786  0.9863
8       968     9       6       9017    0.9908  0.9938  0.9923
9       999     22      10      8969    0.9785  0.9901  0.9842

宏平均 (Macro-Average):
  精确率: 0.9893 (98.93%)
  召回率: 0.9891 (98.91%)
  F1分数: 0.9892

微平均 (Micro-Average):
  精确率: 0.9892 (98.92%)
  召回率: 0.9892 (98.92%)
  F1分数: 0.9892

总体准确率: 0.9892 (98.92%)

分析说明：
  宏平均和微平均接近，说明各类别样本量相对均衡
"""

import pickle

import Base_Classes
import classes
import numpy as np
import support
from feature_extractor import extract_features_batch

# 定义学习率
lr = 0.005  # Adam 通常使用较小的学习率
# 定义学习轮数
epoch = 20
# 定义batch_size
batch_size = 64


# 准备数据
path = "d:\\Machine-Intelligence\\problem_1\\DATA\\MNIST"
train_images, train_label, test_images, test_labels = support.load_mnist(path)

train_features = extract_features_batch(
    train_images, save_scaler_if_needed=True
)
test_features = extract_features_batch(test_images)
input_dims = train_features.shape[1]
# 定义网络
# 层的顺序: Linear -> BatchNorm -> ReLU
# BatchNorm 对未激活的值进行归一化
net = Base_Classes.Sequential(
    [
        # 第一层: 1439-> 512
        classes.Linear(
            input_dims, 512, bias=True, weight_init="kaiming_normal"
        ),
        classes.Batchnorm(num_features=512),  # BatchNorm: 归一化
        classes.Dropout(p=0.1),
        classes.ReLU(),  # ReLU: 激活
        # 第二层:512->256
        classes.Linear(512, 256, bias=True, weight_init="kaiming_normal"),
        classes.Batchnorm(num_features=256),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活
        # 第三层:256->128
        classes.Linear(256, 128, bias=True, weight_init="kaiming_normal"),
        classes.Batchnorm(num_features=128),  # BatchNorm: 归一化
        classes.Dropout(p=0.1),
        classes.ReLU(),  # ReLU: 激活
        # 第四层: 128 -> 64
        classes.Linear(128, 64, bias=True, weight_init="kaiming_normal"),
        classes.Batchnorm(num_features=64),  # BatchNorm: 归一化
        classes.Dropout(p=0.1),
        classes.ReLU(),  # ReLU: 激活
        # 输出层:64->10
        classes.Linear(64, 10, bias=True, weight_init="kaiming_normal"),
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
    print(f"完成第{training_times + 1}轮训练，平均损失为{average_loss}")
    # 设置为预测模式
    net.eval()
    prediction = net(test_features)
    final_prediction = np.argmax(prediction, axis=1)
    correct = np.sum(final_prediction == test_labels)
    total_test_num = len(test_labels)
    accuracy = correct / total_test_num
    print(f"第{training_times + 1}次训练后，在测试集上,正确率为{accuracy}")


# 评估
prediction = net(test_features)
final_prediction = np.argmax(prediction, axis=1)
support.compute_overall_metrics(test_labels, final_prediction)
# 保存模型
model_path = "MLP_1.pkl"
with open(model_path, "wb") as f:
    pickle.dump(net, f)
print(f"[OK] Model saved to: {model_path}")
