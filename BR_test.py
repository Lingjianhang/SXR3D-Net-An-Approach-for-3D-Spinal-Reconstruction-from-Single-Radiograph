import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torchvision.transforms as transforms
from PIL import Image

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

    def forward(self, x_1, x_2):
        x1 = self.encoder1_1(x_1)  # 获取512*8*8特征图
        x1_apfeature = self.encoder1_2(x1)  # 获取特征向量512*2*2
        x2 = self.encoder2(x_2)  # 特征图512*8*8（这里发现反卷积成完整图再反卷积回去是一样的）
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
        for inputs1, inputs2, targets in (dataloader):
            to_generatordata1 = inputs1
            to_generatordata1 = to_generatordata1.to(torch.float32)
            to_generatordata1 = torch.unsqueeze(to_generatordata1, 1).cuda()
            to_generatordata1 = torch.div(to_generatordata1, 255)  # 归一化

            to_generatordata2 = inputs2
            to_generatordata2 = to_generatordata2.to(torch.float32)
            to_generatordata2 = torch.unsqueeze(to_generatordata2, 1).cuda()
            to_generatordata2 = torch.div(to_generatordata2, 255)  # 归一化

            real_data = targets
            real_data = real_data.to(torch.float32)
            real_data = torch.unsqueeze(real_data, 1).cuda()

            fake_data = generator(to_generatordata1, to_generatordata2)

            # 更新生成器
            generator.zero_grad()
            loss_g = criterion(fake_data, real_data)
            loss_g.backward()
            optimizer_g.step()
            i = i + 1
            if i % 450 == 0:
                print('[%d/%d][%d/%d]  Loss_G: %.4f'
                      % (epoch, num_epochs, i, 1188, loss_g.item()))
            if i == 1350:
                scheduler.step()
                print("lr={}".format(optimizer_g.state_dict()['param_groups'][0]['lr']))
                break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
n = 'L'
i = 1
gen.load_state_dict(torch.load('C:\\Users\\Administrator\\single_'+n+str(i)+'BR.pth', map_location=device))
filename1 = r'D:\coronal_scoliosis_4_360\dataset\train\x1\12000.jpg'
filename2 = r'D:\coronal_scoliosis_4_360\dataset\train\x2\12000.jpg'
img1 = Image.open(filename1)
img2 = Image.open(filename2)
# 定义图像变换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
img_tensor1 = transform(img1).to(device)
# 将输入张量的维度扩展一维，转换为 4D 张量
img_tensor1 = torch.unsqueeze(img_tensor1, dim=0)

img_tensor2 = transform(img2).to(device)
# 将输入张量的维度扩展一维，转换为 4D 张量
img_tensor2 = torch.unsqueeze(img_tensor2, dim=0)

with torch.no_grad():
    generated_images = gen(img_tensor1,img_tensor1)
generated_images = torch.where(generated_images < 0.5, torch.tensor(0), torch.tensor(1))


x = generated_images.cpu()

# 将张量数据转换为 Numpy 数组
x_np = x.numpy()[0, 0, :, :, :]
data = x_np[:,:,:] #注意分割 保留部分相同部分

import numpy as np
import pyvista as pv

# 创建三维体数据
data = x_np

# 创建点云数据
points = np.argwhere(data)

# 创建点云对象
cloud = pv.PolyData(points)

# 绘制点云图
plotter = pv.Plotter()
plotter.add_mesh(cloud, point_size=10, render_points_as_spheres=True)
plotter.show()
