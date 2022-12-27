from torch import nn
import torch.nn.functional as F
import torch
from time import time

class Encoder(nn.Module):
    def __init__(self, in_channels=1, h=256, dropout=0.5, args=None):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, args.o_channels_l1, args.kernel_l1, stride=1, padding=args.padding_l1) #65 入力サイズ128/2+1が65
        self.conv2 = nn.Conv2d(args.o_channels_l1, args.o_channels_l2, args.kernel_l2, stride=1, padding=args.padding_l2) #33 入力サイズ65/2+1が33
        self.conv3 = nn.Conv2d(args.o_channels_l2, args.o_channels_l3, args.kernel_l3, stride=1, padding=args.padding_l3) #17

        self.bn1 = nn.BatchNorm2d(args.o_channels_l1)
        self.bn2 = nn.BatchNorm2d(args.o_channels_l2)
        self.bn3 = nn.BatchNorm2d(args.o_channels_l3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        #self.avgpool = nn.AvgPool2d(kernel_size=5)
        #self.fc = nn.Linear(50, 512)

        self.in_linear = (args.o_channels_l3 * ((args.size - (args.kernel_l1 - 1) - (args.kernel_l2 - 1) - (args.kernel_l3 - 1) + (args.padding_l1 * 2) + (args.padding_l2 * 2) + (args.padding_l3 * 2)) ** 2))

        # size 32
        self.fc = nn.Linear(self.in_linear, 512) # for mnist


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # torch.cuda.synchronize()
        # start = time()
        bs = x.size(0)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # torch.cuda.synchronize()
        # elapsed_time = time() - start
        x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc(x)
        return x
        # return x, elapsed_time


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.l1(x)
        return x


class CNNEx(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, target=False,args=None):
        super(CNNEx, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, args=args)
        self.classifier = Classifier(n_classes)
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
