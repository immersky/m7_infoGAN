个人学习infoGAN
infoGAN由于概率论知识有所遗忘和学习范围的覆盖程度不够，理解的不够透彻...其次这次目前我还没有推导整个模型中间特征图的全部变换


InfoGan论文:https://arxiv.org/pdf/1606.03657.pdf

参考:https://blog.csdn.net/z704630835/article/details/83211086

**https://blog.csdn.net/u011699990/article/details/71599067

https://zhuanlan.zhihu.com/p/73324607

https://blog.csdn.net/qq_43827595/article/details/121537291

所有GAN：https://github.com/pianomania/infoGAN-pytorch

学习完GAN和DCGAN还不够，因为他们有时候生成不出来特定数据，那么就需要学习一下CGAN或infoGAN,CGAN是在GAN的基础上加入了标签再训练，且是有监督学习，而infoGAN无监督

先简单看一下CGAN

CGAN 论文 https://arxiv.org/pdf/1411.1784.pdf

CGAN：将原始数据打上标签，将噪声z和标签y拼接在一起，然后作为生成器的输入，在生成MINIST的任务中，作者做法是，先生成100维度的均匀分布的噪声和one-hot类别标签y，然后将这两个数据分别映射到隐层（200、1000），然后将其拼接成一个1200维的向量，最后通过输出层输出784维的图片，对于Discriminator，则是将G生成的图片和图片标签作为两个隐藏层的输入，然后再将两个隐藏层的输出拼接再一起作为输出层的输入。

InfoGAN:通过非监督学习得到可分解的特征表示,作者在DCGAN生成器中除了原先的噪声z还增加了一个隐含编码c，提出了一个新的GAN模型—InfoGAN。Info代表互信息，它表示生成数据x与隐藏编码c之间关联程度的大小，为了使的x与c之间关联密切，所以我们需要最大化互信息的值，据此对原始GAN模型的值函数做了一点修改，相当于加了一个互信息的正则化项。是一个超参，通过之后的实验选择了一个最优值1。

结构:

![struction](https://user-images.githubusercontent.com/74494790/171648323-3103b7d1-7495-45fc-8b37-b86fb9849b19.jpg)


目标:


![mubiao](https://user-images.githubusercontent.com/74494790/171648346-2062ebd8-6959-459d-817e-f5a95e3a6f70.jpg)



当使用MINST数据集，10个数字对应10个特征时，结构如下

![moxing](https://user-images.githubusercontent.com/74494790/171648361-5e4f9b5a-21ab-4772-b4dd-6e884b9e244d.png)




在InfoGAN中输入Generator的噪音z分成了两个部分：一部分是随机噪音z’，另一部分是由若干个隐向量拼接而成latent code c。

c里面的每个维度符合先验的概率分布，比如$categorical\ code\  c_1\sim Cat(K=10,p=0.1)$，two continuous codes $c 2 , c 3 ∼ U n i f ( − 1 , 1 ) $

个人理解，我们相当于给G输入高斯噪音，以及上述几个满足不同分布的c1,c2,c3，G生成数据后，再给D（这里的D包括图中的Classifier和Discriminator，两者网路前段部分一体，在最后一层分开，分别预测c和真伪）判断真伪和C，训练迭代时，也分别有三个部分，训练G生成数据骗过D，训练D分辨G生成的数据，训练D还原用来生成G的c,最终，c按照其给定的分布将映射到数据某特征的分布，很明显，论文提到，c1的10个状态最终对应上了10个数字，我推测是因为c1因为服从的分布对应上了MINST10个数据的分布。

注：一般来说，服从Categorical Distribution的变量都是一个向量，并且是一个One-hot编码的形式，所以c1为10维 one-hot向量，one-hot就是用N位状态寄存器编码N个状态，每个状态都有独立的寄存器位，且这些寄存器位中只有一位有效。



推导：

转载自https://blog.csdn.net/qq_43827595/article/details/121537291

![t1](https://user-images.githubusercontent.com/74494790/171648402-256592d3-c889-4ee6-b451-8cfa1c598a28.png)

![t2](https://user-images.githubusercontent.com/74494790/171648418-4c8a7413-fe07-4b4d-a58d-7814fb91ef36.png)


# 示例

转载自https://blog.csdn.net/m0_62128864/article/details/123997832



CGAN论文里的图示


![cgan](https://user-images.githubusercontent.com/74494790/171648486-94b0e20a-c836-46dc-9623-6f50429d9229.png)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils import data
import os
import glob
from PIL import Image
 
# 独热编码
# 输入x代表默认的torchvision返回的类比值，class_count类别值为10
def one_hot(x, class_count=10):
    return torch.eye(class_count)[x, :]  # 切片选取，第一维选取第x个，第二维全要
 
 
transform =transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)])
 
dataset = torchvision.datasets.MNIST('data',
                                     train=True,
                                     transform=transform,
                                     target_transform=one_hot,
                                     download=False)
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)
 
 
# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(10, 128 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(128 * 7 * 7)
        self.linear2 = nn.Linear(100, 128 * 7 * 7)
        self.bn2 = nn.BatchNorm1d(128 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128,
                                          kernel_size=(3, 3),
                                          padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1,
                                          kernel_size=(4, 4),
                                          stride=2,
                                          padding=1)
 
    def forward(self, x1, x2):
        x1 = F.relu(self.linear1(x1))
        x1 = self.bn1(x1)
        x1 = x1.view(-1, 128, 7, 7)
        x2 = F.relu(self.linear2(x2))
        x2 = self.bn2(x2)
        x2 = x2.view(-1, 128, 7, 7)
        x = torch.cat([x1, x2], axis=1)
        x = F.relu(self.deconv1(x))
        x = self.bn3(x)
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.tanh(self.deconv3(x))
        return x
 
# 定义判别器
# input:1，28，28的图片以及长度为10的condition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(10, 1*28*28)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*6*6, 1) # 输出一个概率值
 
    def forward(self, x1, x2):
        x1 =F.leaky_relu(self.linear(x1))
        x1 = x1.view(-1, 1, 28, 28)
        x = torch.cat([x1, x2], axis=1)
        x = F.dropout2d(F.leaky_relu(self.conv1(x)))
        x = F.dropout2d(F.leaky_relu(self.conv2(x)))
        x = self.bn(x)
        x = x.view(-1, 128*6*6)
        x = torch.sigmoid(self.fc(x))
        return x
 
# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
 
# 损失计算函数
loss_function = torch.nn.BCELoss()
 
# 定义优化器
d_optim = torch.optim.Adam(dis.parameters(), lr=1e-5)
g_optim = torch.optim.Adam(gen.parameters(), lr=1e-4)
 
 
# 定义可视化函数
def generate_and_save_images(model, epoch, label_input, noise_input):
    predictions = np.squeeze(model(label_input, noise_input).cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2, cmap='gray')
        plt.axis("off")
    plt.show()
noise_seed = torch.randn(16, 100, device=device)
 
label_seed = torch.randint(0, 10, size=(16,))
label_seed_onehot = one_hot(label_seed).to(device)
print(label_seed)
# print(label_seed_onehot)
 
# 开始训练
D_loss = []
G_loss = []
# 训练循环
for epoch in range(150):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader.dataset)
    # 对全部的数据集做一次迭代
    for step, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)
        size = img.shape[0]
        random_noise = torch.randn(size, 100, device=device)
 
        d_optim.zero_grad()
 
        real_output = dis(label, img)
        d_real_loss = loss_function(real_output,
                                    torch.ones_like(real_output, device=device)
                                    )
        d_real_loss.backward() #求解梯度
 
        # 得到判别器在生成图像上的损失
        gen_img = gen(label,random_noise)
        fake_output = dis(label, gen_img.detach())  # 判别器输入生成的图片，f_o是对生成图片的预测结果
        d_fake_loss = loss_function(fake_output,
                                    torch.zeros_like(fake_output, device=device))
        d_fake_loss.backward()
 
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()  # 优化
 
        # 得到生成器的损失
        g_optim.zero_grad()
        fake_output = dis(label, gen_img)
        g_loss = loss_function(fake_output,
                               torch.ones_like(fake_output, device=device))
        g_loss.backward()
        g_optim.step()
 
        with torch.no_grad():
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        if epoch % 10 == 0:
            print('Epoch:', epoch)
            generate_and_save_images(gen, epoch, label_seed_onehot, noise_seed)
```

其它的GAN结构
![allgan](https://user-images.githubusercontent.com/74494790/171648534-3eb78385-7c77-4ff1-9ff9-68b56cc34025.png)

