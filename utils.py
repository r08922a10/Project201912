import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

def load_data(file_name="data.csv"):
    
    df = pd.read_csv(file_name)
    
    header_x = ['Spherical Degree', 'Cylindrical Degree', 'Axial', 'Brand', 'FK', 'TP', 'DIA', 'Flat K', 'Steep K', '∆K', 'Flat e', 'Steep e', 'Emeam', 'BFS', 'Sag Differential at 8 mm', 'F Weighted AVE']
    
    
    #header_x = ['Spherical Degree', 'Cylindrical Degree', 'Brand', 'FK', 'TP', 'DIA', 'Steep K', '∆K', 'Emeam', 'BFS', 'Sag Differential at 8 mm', 'F Weighted AVE']
    
    header_y = ["Brand", "FK", "TP", "DIA"]
    for y in header_y:
        header_x.remove(y)
    x = df[header_x].to_numpy()
    y = df[header_y].to_numpy()
    y[y=="ED"] = 0
    y[y=="DL"] = 1
    
    for i in range(len(x)):
        
        for j in range(len(x[i])):
            x[i][j] = float(x[i][j])
    for i in range(len(y)):
        for j in range(len(y[i])):
            y[i][j] = float(y[i][j])
    a = 0
    b = 0
    c = 0
    
    for i in range(len(x)):
        flat_k = int(x[i][3]) + np.around(( x[i][3]-int(x[i][3]) )/0.125)*0.125 #int(predict_y[i]) + np.around( ( predict_y[i]-int(predict_y[i]) )/0.125 )*0.125
        FK = y[i][1]
        #FK = int(FK) + np.around(( FK-int(FK) )/0.125)*0.125
        TP = x[i][0] + 0.5*x[i][1]
        DIA = y[i][3]
        if abs(TP - y[i][2]) <=0.125:
            b+=1
        #else:
            #print(TP, y[i][2])
        y[i][1] = x[i][3] - y[i][1]
        #y[i][3] = 10.6 - y[i][3]
        if abs(flat_k - FK) <=0.125:
            a+=1
        if abs(DIA - 10.6) <=0.2:
            c+=1
            
    y = y.astype(np.float)
    '''
    print(a/len(y))
    print(b/len(y))
    print(c/len(y))
    print(np.sum(y[:,0])/len(y))
    print(y.shape)
    '''
    #exit()
    LR = LinearRegression()
    LR.fit(x, y[:,1])
    #print(LR.score(x,y[:,1]))
    #print(LR.coef_)
    DT = DecisionTreeRegressor()
    DT.fit(x, y[:, 1])
    #print(sum(y - clf.predict(x)))
    #print(clf.predict(x)[:5])
    #print(y[:5])
    impt = DT.feature_importances_
    indices = np.argsort(impt)[::-1]
    '''
    print(DT.feature_importances_)
    print(indices)
    print(np.array(header_x)[indices])
    '''
    #exit()
    valid_split = 0.2 #0.1
    seed = 87
    indices = list(range(len(x)))
    split = int(np.floor(valid_split * len(x)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_indices, valid_indices = indices[split:], indices[:split]
    train_x, train_y = x[train_indices], y[train_indices]
    valid_x, valid_y = x[valid_indices], y[valid_indices]
    max_acc = 0
    '''
    for d in range(1, 5):
        for n in range(1, 200):
            gbr = GradientBoostingClassifier(n_estimators=n, max_depth=d, min_samples_split=2, learning_rate=0.1, verbose=0, validation_fraction=0)
            gbr.fit(train_x, train_y[:, 0])
            if gbr.score(valid_x, valid_y[:, 0]) > max_acc:
                max_acc = gbr.score(valid_x, valid_y[:, 0])
                print(n, d, gbr.score(train_x, train_y[:, 0]), gbr.score(valid_x, valid_y[:, 0]))
    exit()
    '''
    gbr = GradientBoostingClassifier(n_estimators=14, max_depth=3, min_samples_split=2, learning_rate=0.1, verbose=0, validation_fraction=0.1)
    gbr.fit(train_x, train_y[:, 0])
    '''
    print(gbr.score(train_x, train_y[:, 0]), gbr.score(valid_x, valid_y[:, 0]), gbr.score(x, y[:, 0]))
   # print(gbr.predict(train_x)[:5], train_y[:5, 0])
    print(confusion_matrix(gbr.predict(train_x), train_y[:, 0]))
    print(confusion_matrix(gbr.predict(valid_x), valid_y[:, 0]))
    print(confusion_matrix(gbr.predict(x), y[:, 0]))
    impt = gbr.feature_importances_
    #print(impt)
    #exit()
    indices = np.argsort(impt)[::-1]
    print(np.array(header_x)[indices])
    '''
    return x, y



class TrainingDataset(Dataset):
    def __init__(self, normalize=False):
        x, y = load_data()
        if normalize:
            self.x_mean = np.mean(x, axis=0)
            self.x_std = np.std(x, axis=0)
            x = (x-self.x_mean)/self.x_std
        self.x = x
        self.y = y#reshape(-1, 1)
        self.l = len(self.x)

        #print(x_std)

        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.l
#load_data()
