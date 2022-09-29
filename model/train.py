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
#from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error 


def train_Net(logger, data, METRIC, MODE, batch_size, lr, epochs, hidden_dim):
# # def train_Net(logger, data, METRIC, MODE, batch_size, lr, epochs, hidden_dim, group_dim, Q_NUM, dot, EX_NUM=4, lamb=0.1):

    df_pred = pd.DataFrame(columns=("METRIC", "r2",  "MSE"))

    k_r2 = 0
    k_MSE = 0
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
    #print(Y)

  

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1004) #train set, test set 나누기
      

    # TODO: scale
    # print(X_tr)
    # quit()
    scaler_X = MinMaxScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr) #scale 값

    #적용
    norm_X_tr = torch.Tensor(scaler_X.transform(X_tr)).cuda()
    norm_X_te = torch.Tensor(scaler_X.transform(X_te)).cuda()
    norm_y_tr = torch.Tensor(scaler_y.transform(y_tr)).cuda()
    norm_y_te = torch.Tensor(scaler_y.transform(y_te)).cuda()


# train 시작
    model = NeuralModel(logger, mode=MODE, batch_size=batch_size, lr=lr, epochs=epochs, 
                            input_dim=norm_X_tr.shape[-1], hidden_dim=hidden_dim, output_dim=norm_y_tr.shape[-1]
                            )
    X = (norm_X_tr, norm_X_te)
    y = (norm_y_tr, norm_y_te)
    model.fit(X, y) #훈련
    outputs = model.predict(norm_X_te) #norm x_te에 대한 모델의 예측값 pred_y_te

    true = norm_y_te.cpu().detach().numpy().squeeze() # (,3) [[1,2,3]] --> [1,2,3] #텐서-gpu에서 cpu로
    pred = outputs.cpu().detach().numpy().squeeze()
#성능계산
    r2_res = r2_score(true, pred)
    MSE_res = mean_squared_error(true, pred)

    cnt += 1
    # print(f"-------{cnt}-------")
    print(f"--------results-------")
    print(f"r2  score = {r2_res:.4f}")
    print(f"MSE score = {MSE_res:.4f}")

#    #k-fold 없어도 됨
#     k_r2 += r2_res
#     k_pcc += pcc_res
#     k_ci += ci_res
#     k_MSE += MSE_res
 #메모리 비우기
    del norm_X_tr, norm_X_te, norm_y_tr, norm_y_te
    torch.cuda.empty_cache()
    
    print(f"-------Mean of results-------")
    print(f"r2  is {k_r2/cnt}")
    print(f"MSE is {MSE_res/cnt}")
    
    # score = [ (METRIC, k_r2/cnt, k_pcc/cnt, k_ci/cnt, MSE_res/cnt) ]
    #주피터
    # score = [ (METRIC, r2_res/cnt, pcc_res/cnt, ci_res/cnt, MSE_res/cnt) ]
    # ex = pd.DataFrame(score, columns=["METRIC", "r2", 'pcc', "ci", "MSE"])
    # df_pred = pd.concat(([df_pred, ex]), ignore_index=True )
    # # return k_r2/cnt, k_pcc/cnt, k_ci/cnt, MSE_res/cnt, true, pred, df_pred
    return r2_res/cnt, MSE_res/cnt, true, pred, df_pred