import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pandas as pd
import os

folder_path = './OpenPack'
files = os.listdir(folder_path)
merged_data = pd.DataFrame()
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        merged_data = merged_data.append(data, ignore_index=True)
train_data = merged_data
x_train = train_data.iloc[:,2:]
y_train = train_data.iloc[:, 1]

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

x_train_tensor = torch.tensor(x_train.values)
y_train_tensor = torch.tensor(y_train)
y_train_tensor = y_train_tensor.to(torch.long)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #x = x.float()
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y


# 定义超参数
input_size = 40  # 输入大小，根据数据集中每个样本的特征数确定
hidden_size = 128  # 隐层大小
num_classes = 11  # 类别数量
batch_size = 64  # 批量大小
learning_rate = 0.001  # 学习率
num_epochs = 10  # 训练轮数

custom_dataset = CustomDataset(x_train_tensor, y_train_tensor)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = MLP(input_size, hidden_size, num_classes)
#model = model.float()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(data_loader):
        # 前向传播
        labels = labels.to(torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))