import numpy as np
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import accuracy_score
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        
        
        self.w = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1),
            #nn.Sigmoid()
            
        )
        
    def forward(self, input):
        
        output = self.w(input)

        
        #print(input.size(), output.size())
        #print(output[0:5])
        #predict = torch.sigmoid(output)
        #print(predict)
        #exit()
        
        return output
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
        out = predict.detach().numpy().reshape(-1,)
        fout = open("brand.csv", "w")
        
        for j in range(449):
            print(str(j+1)+","+str(out[j]), file=fout)
        #print(predict.detach().numpy().reshape(-1,))
        exit()
        y = y[:, 0].view(-1, 1)
        loss = loss_criteria(predict, y)
        l = loss.detach().item()
        a = acc(predict, y)
    model.train()
    return loss, a
def acc(predict, y):
    y = y.detach().numpy().astype(np.int32)
    predict = predict.detach().numpy()
    #print(predict[:5])
    predict[predict>=0] = 1
    predict[predict<0] = 0
    #print(predict[:5])
    #exit()
    return accuracy_score(y, predict)
    
def main():
    data = TrainingDataset(normalize=True)
    valid_split = 0.2 #0.1
    seed = 87
    indices = list(range(len(data)))
    all_indices = list(range(len(data)))
    split = int(np.floor(valid_split * len(data)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_indices, valid_indices = indices[split:], indices[:split]
    #print(train_indices)
    #print(valid_indices)
    #exit(0)
    all_sampler = SequentialSampler(all_indices)
    train_sampler = SequentialSampler(train_indices)
    valid_sampler = SequentialSampler(valid_indices)
    
    train_loader = DataLoader(data, batch_size=len(train_indices), sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(data, batch_size=len(valid_indices), sampler=valid_sampler, num_workers=0)
    all_loader = DataLoader(data, batch_size=len(all_indices), sampler=all_sampler, num_workers=0)
    
    model = LinearModel(data.x.shape[1])#.to(device)
    model.load_state_dict(torch.load("model_log_nn_2_966.pkl"))
    #model_optim = optim.Adagrad(model.parameters(), lr=0.1)
    model_optim = optim.Adam(model.parameters(), lr=0.1)
    loss_criteria = torch.nn.BCEWithLogitsLoss()#BCELoss()
    max_acc = 0
    for epoch in range(100000):
        for i, xy in enumerate(train_loader):
            train_x, train_y = xy
            train_x = train_x.float()#.to(device)
            train_y = train_y.float()#.to(device)
            model_optim.zero_grad()
            predict = model.forward(train_x)
            y = train_y[:, 0].view(-1, 1)

            #exit()
            loss = loss_criteria(predict, y)
            #loss.backward()
            #model_optim.step()
            #valid_loss, valid_acc = valid_eval(valid_loader, loss_criteria, model)
            all_loss, all_acc = valid_eval(all_loader, loss_criteria, model)
            if epoch%1000 == 0:
                
                #print("Loss:", loss.detach().item(), valid_loss.item(), "ACC:", acc(predict, y), valid_acc)
                exit()
                #print(predict[0:5])
                #print(y[0:5])
            
            if valid_acc > max_acc:
                max_acc = valid_acc
                print("Save Model ACC:", valid_acc)
                torch.save(model.state_dict(), "model_log_nn_2.pkl")
              
    
main()