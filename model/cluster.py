from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


import os
import glob
import sys
#config 파일 가져오기 
os.chdir("../data")
sys.path.append('home/sein/ksc_model/data/A-1/configs/A-1')
dir_list = glob.glob('A-1/configs/A-1/*.cnf')
#metrics.csv 파일 가져오기
df = pd.read_csv('A-1/result/external_metrics.csv')
tplt_df = df[['tps','latency_READ']]
print(tplt_df.info())
#print(tplt_df)

#for dir in dir_list:
#   print(pd.read_csv(dir))
#   break

#print(glob.glob('home/stella/ksc_model/data/A-1/configs/A-1/my_*.cnf'))
#print(os.getcwd())
#print(os.path.realpath(cluster.py))
# print(os.path.abspath(cluster.py))
#file_path = 'home/stella/ksc_model/data/A-1/configs/A-1/my_*.cnf'
#print(os.path.splitext(file_path))
#df = pd.read_csv('home/stella/ksc_model/data/A-1/configs/A-1/my_0.cnf')
#print(df)

