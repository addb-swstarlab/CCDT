# from glob import glob
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.cluster import KMeans


import os
# import glob
# import sys
# #config 파일 가져오기 
os.chdir("../data")
# sys.path.append('home/sein/ksc_model/data/A-1/configs/A-1')
# dir_list = glob.glob('A-1/configs/A-1/*.cnf')
# #metrics.csv 파일 가져오기
# df = pd.read_csv('A-1/result/external_metrics.csv')
# tplt_df = df[['tps','latency_READ']]

#plt.scatter(df['tps'],df['latency_READ'])
#plt.xlabel('tps')
#plt.ylabel('latency')
#plt.show()
clus=pd.read_csv("clustering2.csv")
#print(clus.head())

one_hot = pd.get_dummies(clus, columns=['cluster'])
print(one_hot)
