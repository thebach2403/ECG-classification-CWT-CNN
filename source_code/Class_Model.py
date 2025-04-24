import torch.nn.functional as F
from torch.backends import cudnn
import torch.nn as nn
import torch

#Cấu hình PyTorch
cudnn.benchmark = False
cudnn.deterministic = True #Đảm bảo rằng mô hình luôn tạo ra kết quả giống nhau khi huấn luyện lại.

torch.manual_seed(0)

#Xây dựng mô hình CNN
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        #conv1 → conv3: Ba lớp convolutional trích xuất đặc trưng từ ảnh scalogram
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        #bn1 → bn3: Batch normalization giúp mô hình hội tụ nhanh hơn
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        #pooling1 → pooling3: Giảm kích thước feature map
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        #fc1, fc2: Lớp fully connected, ánh xạ đặc trưng vào 4 lớp output
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (16 x 94 x 94)
        x = self.pooling1(x)  # (16 x 18 x 18)
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        x = self.pooling2(x)  # (32 x 5 x 5)
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        x = self.pooling3(x)  # (64 x 1 x 1)

        x = x.view((-1, 64))  # (64,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x
    

