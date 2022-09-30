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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        self.input_dim = input_dim # 22
        self.hidden_dim = hidden_dim # 64
        self.output_dim = output_dim # 1
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())  #hidden layer
        self.im_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Sigmoid()) #output layer
            

    def forward(self, x):
        self.x_kb = self.knob_fc(x)
        self.x_im = self.im_fc(self.x_kb)
        return self.x_im