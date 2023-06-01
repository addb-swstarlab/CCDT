from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from imblearn.over_sampling import SMOTE
from utils import get_logger
from cluster import make_data
from utils import get_filename
from module import *
import argparse
import pandas as pd
import numpy as np
import glob
import os

os.system('clear')

## argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--external', type=str, choices=['TIME', 'RATE', 'WAF', 'SA', 'config0'], help='choose which external matrix be used as performance indicator')
parser.add_argument('--mode', type=str, default='single', help='choose which model be used on fitness function')
parser.add_argument('--hidden_dim', type=int, default=512, help='Define model hidden size')
parser.add_argument('--input_dim', type=int, default=20, help='Define model hidden size')
parser.add_argument('--output_dim', type=int, default=3, help='Define model hidden size')
parser.add_argument('--lr', type=float, default=0.01, help='Define learning rate')  
parser.add_argument('--act_function', type=str, default='Sigmoid', help='choose which model be used on fitness function')   
parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')   
parser.add_argument('--batch_size', type=int, default=64, help='Define model batch size')
parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')
parser.add_argument('--gam', type=float, default=2, help='gamma parameter of focal loss')
parser.add_argument('--alp', type=float, default=0.25, help='alpha parameter of focal loss')


## fix random_seed
args = parser.parse_args()
random_seed = 777
import random
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

if not os.path.exists('logs'):
    os.mkdir('logs')

logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(args):
    logger.info(f'{i}: {vars(args)[i]}')

## load data
clus = pd.read_csv("../data/clustering_new_dataset_5.csv",index_col=0)
data = make_data('../data/configs_new_dataset/5/*.cnf')

## data preprocessing
# one_hot = pd.get_dummies(clus, columns=['cluster'])    
input_data = data
X = input_data#config 파일
Y = clus.iloc[:,2:].astype(int)

# print(X)
# print(Y)
# quit()

# Y = one_hot #one-hot vector 값
# Y = Y.iloc[:,3:]
X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, shuffle=True) #train set, test set 나누기

# norm_X_tr = np.log(X_tr+1)
# norm_X_te = np.log(X_te+1)
scaler_X = MinMaxScaler().fit(X_tr)

norm_X_tr = torch.Tensor(scaler_X.transform(X_tr)).cuda()
norm_X_te = torch.Tensor(scaler_X.transform(X_te)).cuda()

# print("-------------------------------------------")
# for i in range(10):
#     print(X_tr[i])
#     print(norm_X_tr[i])

# quit()
# norm_X_tr, norm_X_te = norm_X_tr.cuda(), norm_X_te.cuda()
y_tr = y_tr.to_numpy()
y_te = y_te.to_numpy()
y_tr = torch.Tensor(y_tr).cuda()
y_te = torch.Tensor(y_te).cuda()
dataset_tr = TensorDataset(norm_X_tr, y_tr)
dataset_te = TensorDataset(norm_X_te, y_te)

## prepare experiment
batch_size = 32
dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
dataloader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)        
best_loss = np.inf #loss 무한대로 선언. 줄여가려고
logger.info(f"[Train MODE] Training Model")       
name = get_filename('model_save', 'model', '.pt')
model = SingleNet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                                output_dim=args.output_dim)  
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
weights = torch.FloatTensor([3/10,5/10,5/10]).cuda()

## number of sample by class(label)
#cluster_0: 1368  >> 7
#cluster_1: 195   >> 1
#cluster_2: 437   >> 2  
# print(model)

## train the model
for epoch in range(args.epochs):
    #loss_tr = self.train(self.model, dataloader_tr)
    total_loss = 0.
    total_acc = 0.
    
    ## start iteration
    for data, target in dataloader_tr:
        
        # initializate optimizer(Adam)
        optimizer.zero_grad()
        
        # convert data type for training model
        data = data.float().cuda()
   
        # calculate output from deep learning model
        output = model(data)
        
        ## calculate loss value by CrossEntropy or FocalLoss 
        # criterion = nn.CrossEntropyLoss(reduction='mean')
        # criterion_focal = FocalLoss_ver2(gamma=args.gam, alpha=args.alp)
        # criterion_focal = FocalLoss_ver2(gamma=args.gam, alpha=args.alp)
        
        # # loss = criterion(output, target)
        # #Focal loss
        # loss = criterion_focal(output, target, weights=weights)
        
        #Cross-Entropy Loss
        target = target.long()
        # print(target)
        # quit()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target.squeeze())
        # print(output)
        # print(target)
        # quit()
        # print(output)
        # print(target)
        # quit()
        #loss = loss*weights    
        
        
        ## backpropagation
        loss.backward()
        
        ## update the model's weight(parameter value)
        optimizer.step()
        
        ## stack(summation) loss value for each epoch
        total_loss += loss.item()
        
        ## calculate accuracy
        pred = torch.argmax(output, 1)
        #true = torch.argmax(target, 1)
        true = target
        # print(pred)
        # print(true)
        # quit()
        # correct_pred = pred == true
        # accuracy = correct_pred.float().mean()
        # total_acc += accuracy 
        correct_pred = pred == target
        accuracy = correct_pred.float().mean()
        total_acc += accuracy 
        
    ## calculte mean of accuracy and loss 
    total_loss /= len(dataloader_tr)
    total_acc /= len(dataloader_tr)
    
    ## check a log (real-time experiment performance)
    if epoch % 10 == 0:
        #print(data)
        
        # print(output)
        # print(torch.mean(model.Layer1[0].weight))
        # print(pred)
        # print(true)
        # #print(correct_pred)
        # print(f"--------results-------")
        # print(f"Loss  = {loss:.4f}")
        # print(f"Accuracy  = {total_acc:.4f}")
        
        
        logger.info(output)
        logger.info(torch.mean(model.Layer1[0].weight))
        logger.info(pred)
        logger.info(true)
        #print(correct_pred)
        logger.info(f"--------results-------")
        logger.info(f"Loss  = {loss:.4f}")
        logger.info(f"Accuracy  = {total_acc:.4f}")
        
        ## validation performance check
        with torch.no_grad():
            for data, target in dataloader_te:
                data = data.float().cuda()
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                # print(output, output.shape)
                # print(target, target.shape)
                # quit()
                # print(target)
                target = target.long()
                loss = criterion(output, target.squeeze())
                true = target.cpu().detach().numpy().squeeze()
                pred = output.cpu().detach().numpy().squeeze()            
                total_loss += loss.item()
                
            total_loss /= len(dataloader_te) 
            # logger.info("Validation_loss : ", total_loss)  
            print("Validation_loss : ", total_loss)  
            
            
true_,pred_ = true.detach().cpu(),pred.detach().cpu()
logger.info(f"Report = {classification_report(true_,pred_)}")     
print(f"Report = {classification_report(true_,pred_)}")         
    
 