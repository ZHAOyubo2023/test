import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# 定义超参数
batch_size = 64
learning_rate = 0.01
epochs = 5

#定义一个列表来收集训练和测试损失
train_losses = []
test_losses = []

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型 定义一个卷积神经网络的类
class Net(nn.Module):
    # 一般在init中定义一个网路的结构
    def __init__(self):
        super(Net, self).__init__()
        
        # 定义卷积层
        # 图像中的通道数(in_channels)1，由卷积产生的通道数(out_channels)32
        # 卷积核的大小（滤波器kernel_size）5，卷积的步数（stride 默认是1）1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # 定义全连接层
        # 这里1024是经过卷积层第二层图像大小*第二层卷积层产生的通道数
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x)) # 全连接层加激活函数
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # 按照维度为1的x进行log_softmax，拿到的是log probability

model = Net()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()  # 累加损失
        loss.backward()
        optimizer.step()
    # 计算平均损失并添加到列表中
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True)  # 获得概率最大的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
    # 计算平均损失并添加到列表中
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练和测试循环
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pt')

# 绘制损失曲线
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='train_loss') #训练损失
plt.plot(test_losses, label='test_loss') #测试损失
plt.title('loss curve')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(frameon=False)
plt.show()

