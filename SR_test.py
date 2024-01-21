import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
            nn.ReLU(),# 改大特征图
        )
        self.encoder1_2 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),   #
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),   #
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
            nn.ConvTranspose3d(2, 16, 3, stride=1, padding=1), #2*32*32*32---8*32*32*32
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 16, 3, stride=2, padding=1, output_padding=1), #8*32*32*32--16*64*64*64
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
        x1 = self.encoder1_1(x)  #获取512*8*8特征图
        x1_apfeature = self.encoder1_2(x1)  #获取生成另一视角特征向量512*2*2
        x2 =  torch.nn.functional.sigmoid(self.decoder2D(x1_apfeature))  #生成另一视角图
        x2 = self.encoder2(x2)   #特征图512*8*8（这里发现反卷积成完整图再反卷积回去是一样的）
        #将512*8*8reshape为32*32*32
        x1 = x1.view(-1, 32, 32, 32)
        x2 = x2.view(-1, 32, 32, 32)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        #这里将两个特征向量拼接，拼接后再生成三维
        x = torch.cat([x1, x2], dim=1) # x.shape = 4232*32*32
        x = self.decoder3D(x)
        x = torch.nn.functional.sigmoid(x)
        return x





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
n = 'L'
i = 1
gen.load_state_dict(torch.load('C:\\Users\\Administrator\\single_'+n+str(i)+'.pth', map_location=device))
filename = r'D:\coronal_scoliosis_4_360\dataset\train\x1\12000.jpg' #35 12000 11724 11506-11590      11696
img = Image.open(filename)

# 定义图像变换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
img_tensor = transform(img).to(device)
# 将输入张量的维度扩展一维，转换为 4D 张量
img_tensor = torch.unsqueeze(img_tensor, dim=0)
with torch.no_grad():
    generated_images = gen(img_tensor)
generated_images = torch.where(generated_images < 0.5, torch.tensor(0), torch.tensor(1))


x = generated_images.cpu()

# 将张量数据转换为 Numpy 数组
x_np = x.numpy()[0, 0, :, :, :]
data = x_np[:,:,:] #注意分割 保留部分相同部分
"""
save_name = 'D:\\Result\\32\\point_cloud'+n+str(i)+'.xyz'
with open(save_name, 'w') as f:
    for x in range(64):
        for y in range(64):
            for z in range(64):
                if data[x][y][z] == 1:
                    f.write(f"{x} {y} {z}\n")
"""


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
"""
# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 获取 x, y, z 坐标轴数z
z, y, x = np.where(x_np > 0)

# 绘制 3D 散点图
ax.scatter(x, y, z , cmap='viridis', marker='.', linewidth=0.5)

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show() 
"""



