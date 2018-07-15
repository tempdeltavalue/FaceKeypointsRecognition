import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5) #224 - 3 + 1
        self.conv1_bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 4) #111 - 3 + 1
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3)  #54 - 3 + 1
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 2) #26 - 3 + 1 6400
        self.conv4_bn = nn.BatchNorm2d(256) 
        
        self.conv5 = nn.Conv2d(256, 512, 1) #12 - 3 + 1
        self.conv5_bn = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512 * 13 * 13, 1500)
        self.fc1_bn = nn.BatchNorm1d(1500)
        self.fc2 = nn.Linear(1500, 700)
        self.fc2_bn = nn.BatchNorm1d(700)
        self.fc3 = nn.Linear(700,136)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
    def forward(self, x):
        
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        x = self.dropout3(x)
        x = x.view(x.size(0),-1)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x