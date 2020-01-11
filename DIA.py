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
            #nn.Dropout(0.1),
            #nn.Linear(input_size, input_size),
            #nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1),
        )
        
    def forward(self, input):
        
        y = self.w(input)#*0 #+ 10.6
        
        
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
        predict = model.forward(x)#*0+10.6
        y = y[:, 3].view(-1, 1)
        loss = loss_criteria(predict, y)
        l = loss.detach().item()
        a = acc(predict, y)
    model.train()
    return loss, a, predict, y

def acc(predict, y):
    y = y.detach().numpy()
    predict = predict.detach().numpy()
    
    

    real_y = y
    predict_y = predict
    correct = 0  
    for i in range(len(y)):
        #print(predict_y[i])
        predict_y[i] = int(predict_y[i]) + np.around( ( predict_y[i]-int(predict_y[i]) )/0.1 )*0.1
        
        if abs(predict_y[i] -  real_y[i]) <= 0.2:
            correct+=1
    #for i in range(5):
        #print(predict_y[i],  real_y[i])
    return correct/len(y)

def error(predict, y):
    y = y.detach().numpy()
    predict = predict.detach().numpy()
    
    label_freq = {}
    label_error = {}

    real_y = y
    predict_y = predict
    correct = 0  
    for i in range(len(y)):
        if y[i][0] not in label_freq:
            label_freq[y[i][0]]=1
        else:
            label_freq[y[i][0]]+=1
            
        if y[i][0] not in label_error:
            label_error[y[i][0]] = 0
        #print(predict_y[i])
        predict_y[i] = int(predict_y[i]) + np.around( ( predict_y[i]-int(predict_y[i]) )/0.1 )*0.1
        
        if abs(predict_y[i] -  real_y[i]) <= 0.2:
            correct+=1
        else:
            if y[i][0] not in label_error:
                label_error[y[i][0]] = 1
            else:
                label_error[y[i][0]] += 1
    #for i in range(5):
        #print(predict_y[i],  real_y[i])
    return label_freq, label_error

    
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
    #model.load_state_dict(torch.load("model/DIA_nn_3.pkl"))
    model.load_state_dict(torch.load("model3_nn_2_966.pkl"))
    #model_optim = optim.Adagrad(model.parameters(), lr=1)
    model_optim = optim.Adam(model.parameters(), lr=0.01)
    loss_criteria = torch.nn.MSELoss()
    max_acc = 0
    for epoch in range(100000):
        for i, xy in enumerate(train_loader):
            train_x, train_y = xy
            train_x = train_x.float()#.to(device)
            train_y = train_y.float()#.to(device)
            model_optim.zero_grad()
            predict = model.forward(train_x)
            y = train_y[:, 3].view(-1, 1)
            #print(predict)
            #print(y)
            loss = loss_criteria(predict, y)
            #loss.backward()
            #model_optim.step()
            all_loss, all_acc, all_predict, all_y = valid_eval(all_loader, loss_criteria, model)
            
            all_predict = all_predict
            out = all_predict.detach().numpy().reshape(-1,)
            fout = open("DIA.csv", "w")
            for j in range(449):
                out[j] = np.around(out[j], 1)
                print(str(j+1)+","+str(out[j]), file=fout)
            
            exit()
            
            valid_loss, valid_acc, valid_predict, valid_y = valid_eval(valid_loader, loss_criteria, model)
            if epoch%1000 == 0:
                
                print("Loss:", loss.detach().item(), valid_loss.item(), "ACC:", acc(predict, y), valid_acc)
                exit()
                label_freq, label_error = error(valid_predict, valid_y)
                
                label_freq = sorted(label_freq.items(), key=lambda d: d[0])
                label_error = sorted(label_error.items(), key=lambda d: d[0])
                #sorted(dict1.items(), key=lambda d: d[1])
                '''
                for i, k in enumerate(label_freq):
                    freq = label_freq[i][1]
                    error_ = label_error[i][1]
                    print(label_freq[i][0], freq, error_)
                exit()
                '''
            if valid_acc > max_acc:
                max_acc = valid_acc
                print("Save Model ACC:", valid_acc)
                torch.save(model.state_dict(), "model/DIA_nn_3.pkl")

            

    
main()