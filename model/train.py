import pandas as pd
import numpy as np
import glob
import torch
from MLP import NeuralModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from cluster import make_data
from sklearn.metrics import log_loss



def train_Net(logger, data, METRIC, MODE, batch_size, lr, epochs, hidden_dim):
# # def train_Net(logger, data, METRIC, MODE, batch_size, lr, epochs, hidden_dim, group_dim, Q_NUM, dot, EX_NUM=4, lamb=0.1):

    df_pred = pd.DataFrame(columns=("METRIC", "r2",  "MSE"))

    k_r2 = 0
    # k_MSE = 0
    cnt = 0




    # dir_list = glob.glob('../data/A-1/configs/A-1/*.cnf')
    
    clus=pd.read_csv("../data/clustering2.csv")
    # rfin_array = np.array(fin_list).astype(float)
    # input_data = torch.tensor(rfin_array)

    one_hot = pd.get_dummies(clus, columns=['cluster'])

    input_data = data

    X = input_data #config 파일
    # print(X)
    # quit()

    Y = one_hot #one-hot vector 값
  
    Y = Y.iloc[:,3:]
    

  

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, shuffle=False) #train set, test set 나누기
      
    #X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1004) #train set, test set 나누기

    # TODO: scale
    # print(X_tr)
    # quit()
    scaler_X = MinMaxScaler().fit(X_tr)
    # scaler_y = StandardScaler().fit(y_tr) #scale 값

    #적용
    norm_X_tr = torch.Tensor(scaler_X.transform(X_tr)).cuda()
    norm_X_te = torch.Tensor(scaler_X.transform(X_te)).cuda()
    # print(y_tr.shape)
    # quit()
    y_tr = y_tr.to_numpy()
    y_te = y_te.to_numpy()
    y_tr = torch.Tensor(y_tr).cuda()
    y_te = torch.Tensor(y_te).cuda()
   
   
    # norm_y_tr = torch.Tensor(scaler_y.transform(y_tr)).cuda()
    # norm_y_te = torch.Tensor(scaler_y.transform(y_te)).cuda()


# train 시작
    # model = NeuralModel(logger, n_estimators = n_estimators, lr=lr, max_depth = max_depth, random_state = random_state
    #                         )
    model = NeuralModel(logger, mode =MODE, batch_size=batch_size, lr=lr, epochs=epochs, 
                            input_dim=norm_X_tr.shape[-1], hidden_dim=hidden_dim, output_dim=y_tr.shape[-1]
                            )
    X = (norm_X_tr, norm_X_te)
    y = (y_tr, y_te)
    model.fit(X, y) #훈련
    outputs = model.predict(norm_X_te) #norm x_te에 대한 모델의 예측값 pred_y_te

    # true = y_te.cpu().detach().numpy().squeeze() # (,3) [[1,2,3]] --> [1,2,3] #텐서-gpu에서 cpu로
    # pred = outputs.cpu().detach().numpy().squeeze()
    true = y_te # (,3) [[1,2,3]] --> [1,2,3] #텐서-gpu에서 cpu로
    pred = outputs
#성능계산 
    pred_2 = pred.detach().cpu().numpy()
    true_2 = true.detach().cpu().numpy()
    # print(pred_2.shape , true_2.shape)
    logloss = log_loss(true_2,pred_2)
    # Ex)  pred : [0.7, 0.2, 0.1]    
    pred = torch.argmax(pred, 1)   # pred 
    # Ex)  pred : [0]
    
    
    # Ex)  true : [1, 0, 0]
    true = torch.argmax(true, 1) 
    # Ex)  true : [0]
    
    # pred == true (??)   >>   [0] == [0]  >> True 
    correct_pred = pred == true
    # correct_pred = [True, False, True,True, False, True,True, False, True,True, False, True,True, False, True]
    # correct_pred.shape : [batch_size]

    accuracy = correct_pred.float().mean()
    # correct_pred.float().sum() >> True가 몇 개 있느냐 
    # correct_pred.float().mean() >> (0+1+0+1+0+1+0+1+0+1+0+1+0+1+0+1+0) / batch_size 
    # print("-------------------------------------")
    # print(type(pred))
    

   

    cnt += 1
    # print(f"-------{cnt}-------")
    print(f"--------results-------")
    # print(f"r2  score = {r2_res:.4f}")
    # print(f"MSE score = {MSE_res:.4f}")
    print(f"Accuracy  = {accuracy:.4f}")
    print(f"Logloss  = {logloss:.4f}")
 #메모리 비우기
    del norm_X_tr, norm_X_te, y_tr, y_te
    torch.cuda.empty_cache()
    
    # print(f"-------Mean of results-------")
    # # print(f"r2  is {k_r2/cnt}")
    # print(f"Accuracy is {MSE_res/cnt}")
    
    return  true, pred, df_pred