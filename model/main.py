import os
import argparse
import pandas as pd
import numpy as np
from train import train_Net
from utils import get_logger
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import glob
#import natsort  # 숫자 정렬용 라이브러

os.system('clear')

parser = argparse.ArgumentParser()
# parser.add_argument('--external', type=str, choices=['TIME', 'RATE', 'WAF', 'SA'], help='choose which external matrix be used as performance indicator')
parser.add_argument('--external', type=str, choices=['TIME', 'RATE', 'WAF', 'SA', 'config0'], help='choose which external matrix be used as performance indicator')

# parser.add_argument('--kf', type=float, default=3, help='Define split number for K-Folds cross validation')
parser.add_argument('--mode', type=str, default='single', help='choose which model be used on fitness function')
parser.add_argument('--hidden_size', type=int, default=16, help='Define model hidden size')
#parser.add_argument('--group_size', type=int, default=32, help='Define model gruop size')
# parser.add_argument('--dot', action='store_true', help='if trigger, model use loss term, dot')
# parser.add_argument('--lamb', type=float, default=0.1, help='define lambda of loss function' )
parser.add_argument('--lr', type=float, default=0.01, help='Define learning rate')  
parser.add_argument('--act_function', type=str, default='Sigmoid', help='choose which model be used on fitness function')   
parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')   
parser.add_argument('--batch_size', type=int, default=64, help='Define model batch size')
parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')




opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')

logger, log_dir = get_logger(os.path.join('./logs'))

## print parser info
logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')




# MODE = 'reshape'
# batch_size = 64
# lr = 0.01
# epochs = 30

# group_dim = 8



def main():

   
    dir_list = glob.glob('../data/A-1/configs/A-1/*.cnf')
    data = dir_list
    # clus=pd.read_csv("../data/clustering2.csv")
    # one_hot = pd.get_dummies(clus, columns=['cluster'])
    if opt.train:
        r2, MSE, true, pred, df_pred = train_Net(logger, data=data, METRIC=opt.external, MODE=opt.mode, 
                                                        batch_size=opt.batch_size, lr=opt.lr, epochs=opt.epochs, 
                                                         hidden_dim=opt.hidden_size) 

        logger.info(f'\npred = \n{pred[:5]}, {np.exp(pred[:5])}')
        logger.info(f'\ntrue = \n{true[:5]}, {np.exp(true[:5])}')
        logger.info(f'\naccuracy(mean_squared_error) = {mean_squared_error(true, pred)}\naccuracy(mean_absolute_error) = {mean_absolute_error(true, pred)}')
        logger.info(f'\nMetric : {opt.external}')
        logger.info(f'  (r2 score) = {r2:.4f}')
        logger.info(f'  (MSE score) = {MSE:.4f}')

            
    elif opt.eval:
        logger.info('## EVAL MODE ##')


if __name__ == '__main__':
    try:
        main()
        print("F")
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
