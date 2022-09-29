# from glob import glob
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.cluster import KMeans
import configparser
import os
import glob
import torch
# import sys
# #config 파일 가져오기 
os.chdir("../data")

properties = configparser.ConfigParser()
# sys.path.append('home/sein/ksc_model/data/A-1/configs/A-1')
dir_list = glob.glob('A-1/configs/A-1/*.cnf')
#mysqld = properties["mysqld"]
k = 0
# print(properties['mysqld'])
# print(properties['mysqld']['log-error'])
fin_list = []
for k in dir_list:
    properties.read(k)
    # print('---------------------------------------------------------------------')
    # print(k)
    # config_dict = {}
    conf_list = []
    for index, i in enumerate(properties['mysqld']):
        if (index != 0)& (index != 1):
            conf_list.append(properties['mysqld'][i]) 
        # print("key: ", i)
        # print("values: ", properties['mysqld'][i])
    # print(conf_list)
    fin_list.append(conf_list)
# print(np.array(fin_list).shape) 
rfin_array = np.array(fin_list)
print(pd.DataFrame(rfin_array))


rfin_array = np.array(fin_list).astype(float)
input_data = torch.tensor(rfin_array)

