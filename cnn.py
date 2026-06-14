import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1):
        # 默认处理灰度图，input_channels = 1，对于breakout，图象210*160
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(input_width * input_height * 64, 16)
        self.fc2 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        
        x1 = self.relu(self.conv1(x))
        x2 = self.maxpool(x1)
        x3 = self.relu(self.conv2(x2))
        x4 = x3.view(-1, x3.size(1) * x3.size(2) * x3.size(3))
        x5 = self.relu(self.fc1(x4))
        x6 = self.relu(self.fc2(x5))
        return x6, x5, x4, x3, x2, x1

    def visualize_one_image(self, x):
        # image should be managed in (channel, height, width)
        if x.ndim == 4:
            # seperate batch
            batch_size = x.size(0)

        for batch_idx in range(batch_size):
            image = x[batch_idx].cpu().numpy()
            plt.show(image)
            plt.pause(0.1)



'''
对于CNN的一些理解:
1. 输出通道的“通道”并不是输入通道的“通道”：
   作为参数的输入通道通常是指灰度/RGB图通道数分别为1，3，这表示输入信号源
   输出通道是指卷积核的数量，个卷积核代表提取出的一种特征（比如：水平/垂直/对角线特征）
'''