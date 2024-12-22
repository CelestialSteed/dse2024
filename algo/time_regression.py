import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
    
class TimeRegression:
    def __init__(self, verbose=False):
        self.model = nn.Linear(1, 1)
        self.verbose = verbose
    def train_time_reg(self, x, time):

        # 归一化输入数据 x 和 target 数据 time
        self.x_mean = x.mean(dim=0)  # 计算 x 的均值
        self.x_std = x.std(dim=0)    # 计算 x 的标准差
        x = (x - self.x_mean) / self.x_std  # 标准化 x

        self.time_mean = time.mean()  # 计算 time 的均值
        self.time_std = time.std()    # 计算 time 的标准差
        time = (time - self.time_mean) / self.time_std  # 标准化 time

        # from sklearn.preprocessing import StandardScaler
        # # 归一化输入数据 x 和 target 数据 time
        # self.scaler_x = StandardScaler()
        # self.scaler_time = StandardScaler()
        # x = torch.tensor(self.scaler_x.fit_transform(x))  # 归一化 x
        # time = torch.tensor(self.scaler_time.fit_transform(time.reshape(-1, 1)))  # 归一化 time, reshaping for compatibility
        
        # 损失函数和优化器
        criterion = nn.MSELoss()  # 均方误差
        optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        epochs = 1000

        if self.verbose:
            print("Training the time prediction model...")
        for epoch in range(epochs):
            # 前向传播
            predictions = self.model(x)
            # 计算损失
            loss = criterion(predictions, time)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每100个epoch打印一次损失
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def evaluate_time(self, x):
        x = (x - self.x_mean) / self.x_std
        predictions = self.model(x).detach()
        predictions = predictions * self.time_std + self.time_mean  # 反归一化 predictions
        return predictions

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 数据：解码宽度 x 和 时间 time
    x = torch.tensor([1, 1, 2, 2, 3, 4], dtype=torch.float32).view(-1, 1)  # 输入 x
    time = torch.tensor([21800, 21790, 28900, 28850, 36000, 43500], dtype=torch.float32).view(-1, 1)  # 输出

    model = TimeRegression(verbose=True)
    model.train_time_reg(x, time)

    predicted_time = model.evaluate_time(x)
    print(predicted_time)

    # 可视化结果
    plt.scatter(x.numpy(), time.numpy(), color='blue', label='data')
    plt.plot(x.numpy(), predicted_time.numpy(), color='red', label='fit')
    plt.xlabel('decode width')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    plt.savefig('time_regression.png')
