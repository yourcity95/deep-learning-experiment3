import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 图片输出尺寸 (1, 28, 28)
        self.C1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),  # C1卷积后尺寸 (6, 24, 24)
            nn.ReLU(),
        )
        self.S2 = nn.MaxPool2d(2)  # S2池化后尺寸 (6, 12, 12)
        self.C3 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),  # C3卷积后尺寸 (16, 8, 8)
            nn.ReLU(),
        )

        self.S4 = nn.MaxPool2d(2)  # S4池化后尺寸 (16, 4, 4)
        self.C5 = nn.Linear(16 * 4 * 4, 120)  # 全连接层输出 (120,)
        self.F6 = nn.Linear(120, 84)  # 全连接层输出 (84,)
        self.out = nn.Linear(84, 10)  # 输出层为10

    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = x.view(x.size(0), -1)   # 将三维tensor转为一维
        x = self.C5(x)
        x = self.F6(x)
        return self.out(x)
