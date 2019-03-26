"""
Easy Mnist tutorial
"""
import torch.nn as nn
import torch.nn.functional as F

class Mnist(nn.Module):
    """Classic Lenet-5 Architecture
    Layer    | Feature Map | Size  | Kernel Size | Stride
    -----------------------------------------------------
      input  |      1      | 32*32 |             |
    Conv1    |      6      | 28*28 |     5*5     |   1
    Maxpool1 |      6      | 14*14 |     2*2     |   2
    Conv2    |     16      | 10*10 |     5*5     |   1
    Maxpool2 |     16      |  5*5  |     2*2     |   2
    Conv3    |    120      |  1*1  |     5*5     |   1
    FC                        84
    FC                        10

    Refernce by : https://engmrk.com/lenet-5-a-classic-cnn-architecture/
    """

    def __init__(self):
        super(Mnist, self).__init__()
        self.conv1 = nn.Sequential( # input 28*28*1
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 14*14*6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5), # 10*10*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 5*5*16
        )
        self.dropout = nn.Dropout2d()

        self.fc = nn.Sequential(
            nn.Linear(320, 50),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.view(-1, 320)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)