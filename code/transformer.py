import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import time
import pickle
from PIL import Image

# 设置随机数种子

# 数据集载入
def DefectDataset(image_path):
    with open(image_path, 'rb') as f:
        stacked_array = pickle.load(f)
    features = [item[0] for item in stacked_array]
    labels = [item[1] for item in stacked_array]
    return np.array(features), np.array(labels)

def ValDefectDataset(image_path):
    with open(image_path, 'rb') as f:
        stacked_array = pickle.load(f)
    return np.array(stacked_array)

# 定义PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=686):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.encoding, mean=0, std=0.1)
        
        
        
    def forward(self, x):
        x = x + self.encoding[:, :x.size(1), :]
        return self.dropout(x)

# 定义Transformer模型
class Transformerclassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hidden_dim, num_layers, dropout):
        super(Transformerclassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # 使用嵌入层
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, 6)  # 全连接层1
        self.init_weights()
        self.fc2 = nn.Linear(6, 4)  # 全连接层2
        self.fc3 = nn.Linear(4, output_dim)  # 全连接层3
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out0 = self.transformer(x)
        out0 = torch.mean(out0, dim=1)
        out1 = torch.relu(self.decoder(out0)) 
        out2 = torch.relu(self.fc2(out1))
        out = self.fc3(out2)
        out = self.sigmoid(out)
        return out
    
# 数据加载与预处理
def TrainDataLoad(image_path):
    features, labels = DefectDataset(image_path)
    X_train, X_test, y_train, y_test = train_test_split(features[:, :686, :], labels, test_size=0.2, random_state=64)
    Xt_train = torch.from_numpy(X_train.astype(np.float32))
    Xt_test = torch.from_numpy(X_test.astype(np.float32))
    yt_train = torch.from_numpy(y_train.astype(np.int64))
    yt_test = torch.from_numpy(y_test.astype(np.int64))
    train_data =TensorDataset(Xt_train,yt_train)
    test_data = TensorDataset(Xt_test,yt_test)
    return train_data, test_data

def ValDataLoad(image_path):
    X_val = ValDefectDataset(image_path)
    Xt_val = torch.from_numpy(X_val[:, :686, :].astype(np.float32))
    return Xt_val


# 初始化模型和损失函数


# 训练模型
def train_model( model, loss_func, optimizer, train_loader, test_loader, val_loader, num_epochs):
    start_time = time.time()  # 计算起始时间
    # 将模型设置为训练模式.
    train_loss_all = []
    results = []
    batch_counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        test_correct = 0

        model.train()
        for step, (inputs, labels) in enumerate (train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        train_loss_all.append(train_loss / len(train_data))
        train_accuracy = 100.0 * train_correct / len(train_data)

        with torch.no_grad():
            model.eval()
            for step, (inputs, labels) in enumerate (test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
            test_accuracy = 100.0 * test_correct / len(test_data)

            batch_counter += 1
            if batch_counter % 20 == 0:
                val_predicted = []
                for step, (inputs) in enumerate (val_loader):
                    inputs = inputs.to(device)
                
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_predicted.append(predicted)
                val_predicted = torch.cat(val_predicted, dim=0)
                array_2d = val_predicted.reshape((100, 170))
                array_2d = array_2d.cpu()
                image = Image.new('L', (array_2d.size(1), array_2d.size(0)))
                pixels = image.load()
                for y in range(array_2d.size(0)):
                    for x in range(array_2d.size(1)):
                        pixel_value = 255 if array_2d[y, x] == 1 else 0
                        pixels[x, y] = pixel_value
                image.save(f'epoch_{epoch}_image.png')

        print('epoch:',epoch, f"Train Loss: {(train_loss / len(train_data)):.6f} | Train Accuracy: {train_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%")
        
        result = {
        'epoch': epoch,
        'Train Loss': (train_loss / len(train_data)),
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        }
        results.append(result)
        time.sleep(0.1)
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    return model, train_loss_all, results

if __name__ == '__main__':
    train_path = 'reconstruct_train_samples.pkl'
    val_path = 'reconstruct_test_samples.pkl'
    train_data, test_data = TrainDataLoad(train_path)
    val_data = ValDataLoad(val_path)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, drop_last=False)
    model = Transformerclassifier(input_dim=9, output_dim=2, num_heads=4, hidden_dim=8, num_layers=1, dropout=0.1)               
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    transformermodel, train_loss_all, results= train_model(model, loss_func, optimizer, train_loader, test_loader, val_loader, num_epochs=20)

    plt.figure(figsize=(10,6))
    plt.plot(train_loss_all,"ro-",label = "Train loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))
    plt.show()

    # 保存模型的状态字典
    torch.save(transformermodel, 'Transformer_model.pkl')
    file_path = 'Transformer_model.csv'  # 指定保存路径
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])
         # 写入每一行结果
        for result in results:
            writer.writerow([result['epoch'], "{:.6f}".format(result['Train Loss']), "{:.2f}%".format(result['Train Accuracy']), "{:.2f}%".format(result['Test Accuracy'])])
