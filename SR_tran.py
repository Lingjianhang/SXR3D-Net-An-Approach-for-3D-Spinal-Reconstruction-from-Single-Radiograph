import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.autograd import Variable
import pickle
from torch.optim import lr_scheduler


# 定义生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1_1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  #
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),  # 改大特征图
        )
        self.encoder1_2 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),  #
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),  #
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.decoder2D = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )
        self.decoder3D = nn.Sequential(
            nn.ConvTranspose3d(2, 16, 3, stride=1, padding=1),  # 2*32*32*32---16*32*32*32
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 16, 3, stride=2, padding=1, output_padding=1),  # 16*32*32*32--16*64*64*64
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 1, stride=1, padding=0, output_padding=0),  # 16*64*64*64---8*64*64*64
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, 1, stride=1, padding=0, output_padding=0),  # 8*64*64*64---1*64*64*64
            nn.BatchNorm3d(1),
            ## 加4层
        )

    def forward(self, x):
        x1 = self.encoder1_1(x)  # 获取512*8*8特征图
        x1_apfeature = self.encoder1_2(x1)  # 获取生成另一视角特征向量512*2*2
        x2 = torch.nn.functional.sigmoid(self.decoder2D(x1_apfeature))  # 生成另一视角图
        x2 = self.encoder2(x2)  # 特征图512*8*8（这里发现反卷积成完整图再反卷积回去是一样的）
        # 将512*8*8reshape为32*32*32
        x1 = x1.view(-1, 32, 32, 32)
        x2 = x2.view(-1, 32, 32, 32)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        # 这里将两个特征向量拼接，拼接后再生成三维
        x = torch.cat([x1, x2], dim=1)  # x.shape = 4232*32*32
        x = self.decoder3D(x)
        x = torch.nn.functional.sigmoid(x)
        return x


# 定义训练函数
def train_gan(generator, dataloader, num_epochs=80, lr=0.00002, beta1=0.9):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 150, 170], gamma=0.5)
    optimizer_g.param_groups[0]['lr'] = lr
    for epoch in range(num_epochs):
        i = 0
        for inputs, targets in (dataloader):
            to_generatordata = inputs
            to_generatordata = to_generatordata.to(torch.float32)
            to_generatordata = torch.unsqueeze(to_generatordata, 1).cuda()
            to_generatordata = torch.div(to_generatordata, 255)  # 归一化

            real_data = targets
            real_data = real_data.to(torch.float32)
            real_data = torch.unsqueeze(real_data, 1).cuda()

            fake_data = generator(to_generatordata)

            # 更新生成器
            generator.zero_grad()
            loss_g = criterion(fake_data, real_data)
            loss_g.backward()
            optimizer_g.step()
            i = i + 1
            if i % 108 == 0:
                print('[%d/%d][%d/%d]  Loss_G: %.4f'
                      % (epoch, num_epochs, i, 1188, loss_g.item()))
            if i == 1188:
                scheduler.step()
                print("lr={}".format(optimizer_g.state_dict()['param_groups'][0]['lr']))
                break


"""============制作数据集合==================="""
import numpy as np
import pickle

save_list = []
filename_x = r'D:\coronal_scoliosis_4_360\dataset\train_1.npy'
filename_y = r'D:\pythonProject5\33L1.npy'
x = np.load(filename_x, allow_pickle=True)
y = np.load(filename_y, allow_pickle=True)
print(y.shape)
print(x.shape)
for i in range(0, 4752):
    j = i * 2 - 1
    combine = (x[j], y[i])
    save_list

# 将列表保存到文件中
with open(r'D:\single_L1_33', 'wb') as f:
    pickle.dump(save_list, f)

"""====================数据读取部分======================"""
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_data = self.data[index][0]
        target_data = self.data[index][1]
        return input_data, target_data


# 读取数据
with open(r'D:\single_L1_33', 'rb') as f:
    my_data = pickle.load(f)

my_dataset = MyDataset(my_data)
my_dataloader = DataLoader(my_dataset, batch_size=4)
"""=============================训练部分==============================="""
generator = Generator().cuda()
train_gan(generator, my_dataloader, num_epochs=300, lr=0.00001, beta1=0.9)
torch.save(generator.state_dict(), 'single_L1_9.pth')
