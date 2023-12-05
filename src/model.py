import torchvision.models as models
import torch.nn as nn
import torch

class Net_50(nn.Module):
    def __init__(self, input_ch=1, num_class=7, pretrained=True):
        super(Net_50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)  # 使用预训练的ResNet-50模型作为基础模型

        # 替换ResNet-50的第一层卷积层
        conv1_weight = torch.mean(self.model.conv1.weight, dim=1, keepdim=True).repeat(1,input_ch,1,1)
        conv1 = nn.Conv2d(input_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 更新模型参数
        model_dict = self.model.state_dict()
        self.model.conv1 = conv1
        model_dict['conv1.weight'] = conv1_weight
        model_dict.update(model_dict)

        # 替换ResNet-50的全连接层
        fc_weight = model_dict['fc.weight']
        fc_weight = fc_weight[:, :3]
        model_dict['fc_weight'] = fc_weight
        model_dict.update(model_dict)
        self.model.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.model(x)
        return x


class Net_18(nn.Module):
    def __init__(self, input_ch=1, num_class=7, pretrained=True):
        super(Net_18,self).__init__()
        self.model = models.resnet18(pretrained=pretrained)  # 使用预训练的ResNet-18模型作为基础模型

        # 替换ResNet-18的第一层卷积层
        conv1_weight = torch.mean(self.model.conv1.weight,dim=1,keepdim=True).repeat(1,input_ch,1,1)
        conv1 = nn.Conv2d(input_ch, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        # 更新模型参数
        model_dict = self.model.state_dict()
        self.model.conv1 = conv1
        model_dict['conv1.weight'] = conv1_weight
        model_dict.update(model_dict)

        # 替换ResNet-18的全连接层
        fc_weight = model_dict['fc.weight']
        fc_weight = fc_weight[:, :3]
        model_dict['fc_weight'] = fc_weight
        model_dict.update(model_dict)
        self.model.fc = nn.Linear(512, num_class)

    def forward(self,x):
        x = self.model(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=10 * 10 * 16, out_features=120)                   
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=7)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)  # 进行第一层卷积和池化操作
        conv2_output = self.conv2(conv1_output)  # 进行第二层卷积和池化操作
        conv2_output = conv2_output.reshape(conv2_output.shape[0], -1)  # 将特征图展平为一维向量
        fc1_output = self.fc1(conv2_output)  # 进行第一层全连接操作
        fc2_output=self.fc2(fc1_output)  # 进行第二层全连接操作
        fc3_output = self.fc3(fc2_output)  # 进行第三层全连接操作
        return fc3_output

