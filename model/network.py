import torch
import torch.nn as nn


class SingleNet(nn.Module):
    # def __init__(self, n_estimators, lr, max_depth, random_state):
    #     super(SingleNet, self).__init__()
    #     self.n_estimators = n_estimators
    #     self.lr = lr
    #     self.max_depth = max_depth                    # learning rate
    #     self.random_state = random_state
    #     self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())  #hidden layer
    #     self.im_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Sigmoid()) #output layer
    # dropout = nn.Dropout(p=0.25)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        # self.input_dim = input_dim # 22
        # self.hidden_dim = hidden_dim
        # self.hidden_dim = hidden_dim # 64
        # self.output_dim = output_dim # 1
        self.input_dim = 20 # 22
        self.hidden_dim = 128
        self.hidden_dim2 = 256
        # self.hidden_dim3 = 128
        self.output_dim = 4 # 64
    
        # self.weight = nn.parameter (torch.FloatTensor(7,32,32, device="cuda"))
        
        
        self.Layer1 = nn.Sequential(nn.BatchNorm1d(self.input_dim),nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())  #hidden layer
        self.Layer2 = nn.Sequential(nn.BatchNorm1d(self.hidden_dim),nn.Linear(self.hidden_dim, self.hidden_dim2), nn.ReLU())
        self.Layer3 = nn.Sequential(nn.BatchNorm1d(self.hidden_dim2),nn.Linear(self.hidden_dim2, self.output_dim), nn.Softmax())
        
        # self.Layer4 = nn.Sequential(nn.LayerNorm(self.hidden_dim3),nn.Linear(self.hidden_dim3, self.output_dim), nn.Softmax())

    def forward(self, x):
       
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        # x = self.Layer4(x)

        # self.x_kb = self.knob_fc(x)
        # self.
        # self.x_im = self.im_fc(self.x_kb)

      
        return x