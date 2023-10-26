import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import AVADataset
from pytorchtools import EarlyStopping
from BayesianRank_loss import BRankLoss
from common import AverageMeter, Transform
import torchvision as tv
import warnings
warnings.filterwarnings("ignore")
base_model = tv.models.resnet50(pretrained=False)
pre = torch.load("resnet50-19c8e357.pth")
base_model.load_state_dict(pre)
class DARN(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(DARN, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.head = nn.Sequential(
            nn.Tanh(inplace=True),
            nn.Linear(inputs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,512),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, outputs),
            nn.Softplus()

            )
        self.b0 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.b3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.b0.data.fill_(-0.75)
        self.b1.data.fill_(-0.25)
        self.b2.data.fill_(0.25)
        self.b3.data.fill_(0.75)

        self.normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    def forward(self, input_1, input_2):
        x1 = self.base_model(input_1) #预测input_1得分
        x1 = x1.view(x1.size(0), -1)
        x1 = self.head(x1)
        x2 = self.base_model(input_2) #预测input_2得分
        x2 = x2.view(x2.size(0), -1)
        x2 = self.head(x2)
        delta_mean = (torch.mean(x1,dim=1)-torch.mean(x2,dim=1)).reshape(-1,1)
        delta_var = ((torch.var(x1,dim=1)+torch.var(x2,dim=1)).reshape(-1,1))**0.5
        b0 = (self.b0+delta_mean)
        b1 = (self.b1+delta_mean)
        b2 = (self.b2+delta_mean)
        b3 = (self.b3+delta_mean)
        p0 = self.normal.cdf(b0.to(device1))
        p1 = self.normal.cdf(b1.to(device1)) - p0
        p2 = self.normal.cdf(b2.to(device1)) - p1 - p0
        p3 = self.normal.cdf(b3.to(device1)) - p2 - p1 - p0
        p4 = 1 - self.normal.cdf(b3.to(device1))
        res = torch.cat([p0,p1,p2,p3,p4],dim=1)
        return res

    def predict(self, input):
        res1 = self.base_model(input)
        res1 = res1.view(res1.size(0),-1)
        res1 = self.head(res1)
        return torch.mean(res1,dim=1)

def get_dataloader(train_csv,vali_csv,path_to_images,batch_size, num_workers):
    transform = Transform()
    train_ds = AVADataset(train_csv, path_to_images, transform.train_transform)
    val_ds = AVADataset(vali_csv, path_to_images, transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader

def train_and_eval(train_path,vali_path,image_path):
    # 超参
    inputs = 2048
    hidden_size = 512
    outputs = 16
    learning_rate = 0.00001
    num_epochs = 200
    batch_size = 64
    model = DARN(inputs, hidden_size, outputs).to(device)
    model = nn.DataParallel(model)
    model = model.cuda()
    #损失函数和优化器
    criterion = BRankLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    early_stopping = EarlyStopping(patience=8,verbose=False) 
    #load dataset
    train_data_loader, vali_data_loader = get_dataloader(train_path, vali_path, image_path, batch_size, 1)
    for epoch in range(num_epochs):
        best_loss = float("inf")
        best_state = None
        train_losses = AverageMeter()
        #train
        model = model.train()
        correct = 0
        total = 0
        for i, (data1, data2, y) in enumerate(train_data_loader):
            data1 = data1.cuda().to(device)
            data2 = data2.cuda().to(device)
            label_size = data1.size()[0]
            total += label_size
            res1 = model(data1, data2)
            label = torch.tensor(y,dtype=torch.float).cuda().to(device)
            #print("res1",res1)
            #print("label",label)
            loss = criterion(res1, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.update(loss.item(),label_size)
            label_y = list(torch.argmax(label,dim=1))
            res_y = list(torch.argmax(res1,dim=1))
            for i in range(label_size):
                if label_y[i]==res_y[i]:correct += 1

        train_acc = correct/total
        #eval
        model = model.eval()
        vali_losses = AverageMeter()
        
        correct = 0
        total = 0
        raw_label = None
        pred_label = None
        for i, (data3,data4,y) in enumerate(vali_data_loader):
            data3 = data3.cuda().to(device)
            data4 = data4.cuda().to(device)
            label_size = data3.size()[0]
            res_1 = model(data3,data4)
            label = torch.tensor(y,dtype=torch.float).cuda().to(device)
            loss = criterion(res_1,label)
            vali_losses.update(loss.item(),label_size)
            label_y = list(torch.argmax(label,dim=1))
            res_y = list(torch.argmax(res_1,dim=1))
            for i in range(label_size):
                if label_y[i] == res_y[i]:correct += 1
            total += label_size
        train_loss = train_losses.avg
        val_loss = vali_losses.avg
        val_acc = correct/total
        print('Epoch [{}/{}], Avg_train_loss: {:.4f} Avg_train_acc: {:.4f} Avg_vali_loss: {:.4f} val_acc: {:.4f}'.format(epoch + 1, num_epochs, train_loss,train_acc,val_loss,val_acc))
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("early stopping")
            break
        if np.isnan(val_loss):
            print("val loss is nan")
            break
        if best_state is None or val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    }
            torch.save(best_state,"./output/best_state_v6_mvCrop_1118.pth")

if __name__ == "__main__":
    import sys
    train_path = sys.argv[1]
    vali_path = sys.argv[2]
    image_path = sys.argv[3]
    train_and_eval(train_path,vali_path, image_path)
