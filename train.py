import model
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn as nn

EPOCH = 5               # 学习次数
BATCH_SIZE = 50         #块大小
LR = 0.01              # 学习率

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # 将验证样本集尺寸从(2000, 28, 28)转为(2000, 1, 28, 28)，值域为(0,1)
test_y = test_data.test_labels[:2000]


net = model.Model()
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=LR)   # 实例化SGD优化器
loss_func = nn.CrossEntropyLoss()                       # 定义loss函数

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # 取每个样本块训练
        b_x = Variable(x)
        b_y = Variable(y)

        output = net(b_x)               # 模型输出
        loss = loss_func(output, b_y)   # 计算损失
        optimizer.zero_grad()           # 清除当前优化器梯度
        loss.backward()                 # 损失回传
        optimizer.step()                # 应用梯度

        if step % 50 == 0:
            test_output = net(test_x)   # 每50次计算分类精度
            pred_y = torch.max(test_output, 1)[1].data.squeeze()    # 验证样本预测值最大的标签
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))    # 计算分类精度
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)  # 输出分类精度
torch.save(net, 'net.pkl')  # 存储模型