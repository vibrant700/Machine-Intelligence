"""
python problem_1/Model_Related_Code/pytoch_style_code/try_features1.py
完成第1轮训练，平均损失为0.14088238616290483
完成第2轮训练，平均损失为0.06682389808114932
完成第3轮训练，平均损失为0.05180469871300587
完成第4轮训练，平均损失为0.039420572114578376
完成第5轮训练，平均损失为0.03462301003926901
完成第6轮训练，平均损失为0.02789542838894281
完成第7轮训练，平均损失为0.024610621745744972
完成第8轮训练，平均损失为0.022719838499853137
完成第9轮训练，平均损失为0.018409439316111272
完成第10轮训练，平均损失为0.0175536538573146
完成第11轮训练，平均损失为0.015925420308524462
完成第12轮训练，平均损失为0.013231521089094785
完成第13轮训练，平均损失为0.012300734431422978
完成第14轮训练，平均损失为0.011964347700112283
完成第15轮训练，平均损失为0.009492396215127116
完成第16轮训练，平均损失为0.010581967885919821
完成第17轮训练，平均损失为0.009053302322328786
完成第18轮训练，平均损失为0.010083170563968406
完成第19轮训练，平均损失为0.007144273016547625
完成第20轮训练，平均损失为0.008717929055622894
0       972     9       8       9011    0.9908  0.9918  0.9913
1       1129    12      6       8853    0.9895  0.9947  0.9921
2       1021    14      11      8954    0.9865  0.9893  0.9879
3       994     12      16      8978    0.9881  0.9842  0.9861
4       972     16      10      9002    0.9838  0.9898  0.9868
5       883     12      9       9096    0.9866  0.9899  0.9882
6       950     12      8       9030    0.9875  0.9916  0.9896
7       1014    14      14      8958    0.9864  0.9864  0.9864
8       956     8       18      9018    0.9917  0.9815  0.9866
9       985     15      24      8976    0.9850  0.9762  0.9806

宏平均 (Macro-Average):
  精确率: 0.9876 (98.76%)
  召回率: 0.9876 (98.76%)
  F1分数: 0.9876

微平均 (Micro-Average):
  精确率: 0.9876 (98.76%)
  召回率: 0.9876 (98.76%)
  F1分数: 0.9876

总体准确率: 0.9876 (98.76%)

分析说明：
  宏平均和微平均接近，说明各类别样本量相对均衡
"""
import Base_Classes
import classes
import numpy as np
import support
from feature_extractor import extract_features_batch
from torch.utils.tensorboard import SummaryWriter

# 定义学习率
lr = 0.005  # Adam 通常使用较小的学习率
# 定义学习轮数
epoch = 20
# 定义batch_size
batch_size = 64


# 准备数据
path = "problem_1/DATA/MNIST"
train_images, train_label, test_images, test_labels = support.load_mnist(
    path
)

train_features = extract_features_batch(train_images, save_scaler_if_needed=True)
test_features = extract_features_batch(test_images)
input_dims = train_features.shape[1]
# 定义网络
# 层的顺序: Linear -> BatchNorm -> ReLU
# BatchNorm 对未激活的值进行归一化
net = Base_Classes.Sequential(
    [
        # 第一层: 784 -> 256
        classes.Linear(input_dims, 512, bias=True),
        classes.Batchnorm(num_features=512),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活

        classes.Linear(512, 256, bias=True),
        classes.Batchnorm(num_features=256),  # BatchNorm: 归一化
        classes.ReLU(),  # ReLU: 激活
        # 第二层: 256 -> 128
        classes.Linear(256, 128, bias=True),
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

# ============ 保存模型 ============
import pickle

# 保存模型
model_path = "problem_1/Model_Related_Code/pytoch_style_code/try_features1_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(net, f)
print(f"[OK] Model saved to: {model_path}")
