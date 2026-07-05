import torch
import torch.nn as nn
import numpy as np
import os
import cv2
class FrameFeatureExtractor(nn.Module):
    def __init__(self, image_size: int, output_dim: int):
        super(FrameFeatureExtractor, self).__init__()
        self.cnn = SimpleCNN(image_size, output_dim)
    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        out = self.cnn(x)
        return out[0]

class SimpleCNN(nn.Module):
    def __init__(self, image_size: int, output_dim: int, input_channels: int = 1):
        # 默认处理灰度图，input_channels = 1，对于breakout，图象84*84
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv1_size = (image_size - 8) // 4 + 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2_size = (self.conv1_size - 4) // 2 + 1
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv3_size = (self.conv2_size - 3 )// 1 + 1
        self.fc1 = nn.Linear(self.conv3_size * self.conv3_size * 64, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        assert(isinstance(x, torch.Tensor))
        if x.ndim == 2:
            x = x.unsqueeze(0)          # -> (step, channel, 84, 84)
        x1 = self.relu(self.conv1(x))   # -> (step, 32, self.conv1_size, self.conv1_size)
        x2 = self.relu(self.conv2(x1))  # -> (step, 64, self.conv2_size, self.conv2_size)
        x3 = self.relu(self.conv3(x2))  # -> (step, 64, self.conv3_size, self.conv3_size)
        x4 = x3.view(-1, self.conv3_size * self.conv3_size * 64)    # -> (step, 64 * self.conv3_size * self.conv3_size)
        x4 = self.relu(self.fc1(x4))    # -> (step, 512)
        x4 = self.relu(self.fc2(x4))    # -> (step, 256)
        return x4, x3, x2, x1

    def visualize_one_image(self, x, title="visualize cnn layer"):
        # image should be managed in (channel, height, width)
        if x.ndim == 4:
            # 一个bacth是一张图片
            x = x[0]

        if x.ndim == 3:
            # channel, height, width
            x = x[0]    # visualize feature channel 0
        
        image = x.cpu().detach().numpy()
        assert(image.ndim == 2)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255.0).astype(np.uint8)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, f"extract_{title}.png")
        cv2.imwrite(save_path, image)




'''
对于CNN的一些理解:
1. 输出通道的“通道”并不是输入通道的“通道”：
   作为参数的输入通道通常是指灰度/RGB图通道数分别为1，3，这表示输入信号源
   输出通道是指卷积核的数量，个卷积核代表提取出的一种特征（比如：水平/垂直/对角线特征）
'''