import torch
import torch.nn as nn

class SingleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        self.input_dim = input_dim # 22
        self.hidden_dim = hidden_dim # 32
        self.output_dim = output_dim # 1
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.Sigmoid())  #hidden layer
        self.im_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim)) #output layer
        

    def forward(self, x):
        self.x_kb = self.knob_fc(x)
        self.x_im = self.im_fc(self.x_kb)
        return self.x_im