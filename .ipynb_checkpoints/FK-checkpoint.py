import numpy as np
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        
        
        self.w = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Linear(input_size*2, input_size*2),
            #nn.ReLU(),
            #nn.Linear(input_size, input_size),
            #nn.ReLU(),
            nn.Linear(input_size, 1),
        )
        
    def forward(self, input):
        
        y = self.w(input)

        
        
        return y
#def evaluate(predict, y):
def valid_eval(valid_loader, loss_criteria, model):
    l = 0
    ac = 0
    model.eval()
    for i, xy in enumerate(valid_loader):
        x, y = xy
        x = x.float()#.to(device)
        y = y.float()#.to(device)
        predict = model.forward(x)

        y = y[:, 1].view(-1, 1)
        loss = loss_criteria(predict, y)
        l = loss.detach().item()
        a = acc(predict, y, x)
    model.train()
    return loss, a
def acc(predict, y, x):
    y = y.detach().numpy()
    predict = predict.detach().numpy()
    x =  x.detach().numpy()

    real_y = x[:, 3].reshape(-1, 1) - y
    predict_y = x[:, 3].reshape(-1, 1) - predict

    correct = 0  
    for i in range(len(y)):

        predict_y[i] = int(predict_y[i]) + np.around( ( predict_y[i]-int(predict_y[i]) )/0.125 )*0.125
        
        if abs(predict_y[i] -  real_y[i]) <= 0.25:
            correct+=1
    
    out = predict_y.reshape(-1,)
    fout = open("FK.csv", "w")
    for j in range(449):
        print(str(j+1)+","+str(out[j]), file=fout)
    exit()
    
    #for i in range(5):
        #print(predict_y[i],  real_y[i])
    return correct/len(y)
    
def main():
    data = TrainingDataset()
    valid_split = 0.2 #0.1
    seed = 87
    indices = list(range(len(data)))
    all_indices = list(range(len(data)))
    split = int(np.floor(valid_split * len(data)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_indices, valid_indices = indices[split:], indices[:split]

    all_sampler = SequentialSampler(all_indices)
    train_sampler = SequentialSampler(train_indices)
    valid_sampler = SequentialSampler(valid_indices)
    
    train_loader = DataLoader(data, batch_size=len(train_indices), sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(data, batch_size=len(valid_indices), sampler=valid_sampler, num_workers=0)
    all_loader = DataLoader(data, batch_size=len(all_indices), sampler=all_sampler, num_workers=0)
    
    model = LinearModel(data.x.shape[1])#.to(device)
    model.load_state_dict(torch.load("model_nn_1_82.pkl"))
    #model_optim = optim.Adagrad(model.parameters(), lr=0.1)
    model_optim = optim.Adam(model.parameters(), lr=0.001)
    loss_criteria = torch.nn.MSELoss()
    max_acc = 0
    for epoch in range(100000):
        for i, xy in enumerate(train_loader):
            train_x, train_y = xy
            train_x = train_x.float()#.to(device)
            train_y = train_y.float()#.to(device)
            model_optim.zero_grad()
            predict = model.forward(train_x)
            y = train_y[:, 1].view(-1, 1)
            #print(predict)
            #print(y)
            loss = loss_criteria(predict, y)
            #loss.backward()
            #model_optim.step()
            all_loss, all_acc = valid_eval(all_loader, loss_criteria, model)
            
            valid_loss, valid_acc = valid_eval(valid_loader, loss_criteria, model)
            
            if epoch%1000 == 0:
                
                print("Loss:", loss.detach().item(), valid_loss.item(), "ACC:", acc(predict, y, train_x), valid_acc)
                exit()
            if valid_acc > max_acc:
                max_acc = valid_acc
                print("Save Model ACC:", valid_acc)
                torch.save(model.state_dict(), "model/FK_nn.pkl")

            

    
main()