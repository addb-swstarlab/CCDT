import os
import utils
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import configparser
from utils import get_logger
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score,precision_score, recall_score, f1_score, roc_auc_score

os.system('clear')

## argument setting
# parser = argparse.ArgumentParser()
#parser.add_argument('--external', type=str, choices=['TIME', 'RATE', 'WAF', 'SA', 'config0'], help='choose which external matrix be used as performance indicator')
#parser.add_argument('--mode', type=str, default='single', help='choose which model be used on fitness function')
#parser.add_argument('--hidden_dim', type=int, default=512, help='Define model hidden size')
#parser.add_argument('--input_dim', type=int, default=20, help='Define model hidden size')
#parser.add_argument('--output_dim', type=int, default=3, help='Define model hidden size')
# parser.add_argument('--lr', type=float, default=0.01, help='Define learning rate')  
# parser.add_argument('--act_function', type=str, default='Sigmoid', help='choose which model be used on fitness function')   
# parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')   
# parser.add_argument('--batch_size', type=int, default=64, help='Define model batch size')
# parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
#parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')
#parser.add_argument('--gam', type=float, default=2, help='gamma parameter of focal loss')
#parser.add_argument('--alp', type=float, default=0.25, help='alpha parameter of focal loss')


## fix random_seed
# args = parser.parse_args()
# # random_seed = 777
# # import random
# # torch.manual_seed(random_seed)
# # torch.cuda.manual_seed(random_seed)
# # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False
# # np.random.seed(random_seed)
# # random.seed(random_seed)

if not os.path.exists('logs'):
    os.mkdir('logs')

logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
# logger.info("## model hyperparameter information ##")
# for i in vars(args):
#     logger.info(f'{i}: {vars(args)[i]}')

## load data
def mysql_knob_dataframe(wk, KNOB_PATH):
    KNOB_PATH = ('../data/configs_new_dataset')
    KNOB_PATH = os.path.join(KNOB_PATH, str(wk))
    config_len = len(os.listdir(KNOB_PATH))
    cnf_parser = configparser.ConfigParser()
    pd_mysql = pd.DataFrame()
    for idx in range(config_len):
        cnf_parser.read(os.path.join(KNOB_PATH, f'my_{idx}.cnf'))
        conf_dict = cnf_parser._sections['mysqld']
        tmp = pd.DataFrame(data=[conf_dict.values()], columns=conf_dict.keys())
        pd_mysql = pd.concat([pd_mysql, tmp])
        
    pd_mysql = pd_mysql.reset_index(drop=True)
    pd_mysql = pd_mysql.drop(columns=['log-error', 'bind-address'])
    return pd_mysql

def KmeanClustering(external, k):
    np_external = external.to_numpy()
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(np_external)
    #labels = kmeans.labels_

def get_data(knob_path, external_path, wk):
    raw_knobs = mysql_knob_dataframe(wk, knob_path)
    external_ = pd.read_csv(os.path.join(external_path, f'external_results_{wk}.csv'), index_col=0)
    latency_columns = []
    for col in external_.columns:
        if col.find("latency") == 0 and col != 'latency_max' and col != 'latency_CLEANUP':
            latency_columns.append(col)
    external = external_[['tps']].copy()
    external['latency'] = external_[latency_columns].max(axis=1)
    return raw_knobs, external

def get_class_num(data):
    cls, cnt = np.unique(data, return_counts=True)
    print(f'# of the largest class / # of data = {max(cnt) / sum(cnt)}')
    for _ in range(len(cls)):
        print(f'{cls[_]} : {cnt[_]}')
        
for wk in range(10):
    KNOB_PATH = ('../data/configs_new_dataset')
    EXTERNAL_PATH=("../data/configs_new_dataset/external_new_dataset/")
    # print(f'==================={wk} workload===================')
    raw_knobs, external = get_data(KNOB_PATH, EXTERNAL_PATH, wk)
    labels = KmeanClustering(external, 3)
    cls, cnt = np.unique(labels, return_counts=True)
    # print(f'{cls[0]} : {cnt[0]:4}\n{cls[1]} : {cnt[1]:4}\n{cls[2]} : {cnt[2]:4}')
    # #print(f'score = {davies_bouldin_score(raw_knobs, labels)}')
    # print(f'==================================================')    
 
 #XGBoost Classifier   
def get_class_num(data):
    cls, cnt = np.unique(data, return_counts=True)
    print(f'# of the largest class / # of data = {max(cnt) / sum(cnt)}')
    for _ in range(len(cls)):
        print(f'{cls[_]} : {cnt[_]}')    
        
for wk in range(10):
    logger.info(f'=WK{wk}===============================================================')
    raw_knobs, external = get_data(KNOB_PATH, EXTERNAL_PATH, wk)
    np_external = external.to_numpy()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(np_external)
    labels = kmeans.labels_
    get_class_num(labels)

    X_train, X_test, y_train, y_test = train_test_split(raw_knobs, labels, test_size=0.4, shuffle=True)
    X_scaler = MinMaxScaler().fit(X_train)

    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    # clf = xgb.XGBClassifier(n_estimators=50, learning_rate = 0.001, max_depth=14, random_state=0)
    #clf = xgb.XGBClassifier(n_estimators=50, learning_rate = 0.001 ,max_depth=14)
    clf = xgb.XGBClassifier(n_estimators=1500, learning_rate = 0.005, 
                            max_depth = 6, max_leaves = 255, n_jobs=-1)
    clf.fit(scaled_X_train, y_train)
    pred = clf.predict(scaled_X_test)
    # print('********true********')
    # print(get_class_num(y_test))
    # print('********pred********')
    # print(get_class_num(pred))
    
    # print('********SCORE********')
    # print('f1 score : ',f1_score(y_test, pred, average='weighted'))
    # print('precision : ',precision_score(y_test, pred, average='weighted'))
    # print('recall : ',recall_score(y_test, pred, average='weighted'))
    # print('accuracy: ',clf.score(scaled_X_test, y_test))
    # print('balanced accuracy: ', balanced_accuracy_score(y_test, pred))     
    
    # logger.info(f"********true********")
    # logger.info(f"{get_class_num(y_test)}")
    # logger.info(f"********pred********")
    # logger.info(f"{get_class_num(pred)}")

    print('********true********')
    print(get_class_num(y_test))
    print('********pred********')
    print(get_class_num(pred))
  
    logger.info(f"********SCORE********")
    logger.info(f"f1 score : {f1_score(y_test, pred, average='weighted')}")
    logger.info(f"precision : {precision_score(y_test, pred, average='weighted')}")
    logger.info(f"recall : {recall_score(y_test, pred, average='weighted')}")
    logger.info(f"accuracy: {clf.score(scaled_X_test, y_test)}")
    logger.info(f"balanced accuracy: {balanced_accuracy_score(y_test, pred)}")      
    
 