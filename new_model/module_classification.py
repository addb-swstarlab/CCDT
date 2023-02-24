import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearModel, self).__init__()
        
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        
        self.conf = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), nn.ReLU())  
        self.cluster = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), nn.ReLU())
        self.attention = nn.MultiheadAttention(self.hidden_dim, 1)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        #self.Layer3 = nn.Sequential(nn.Linear(self.hidden_dim2, self.hidden_dim3), nn.ReLU())
        # self.Layer4 = nn.Linear(self.hidden_dim3, self.output_dim)
        #self.Layer3 = nn.Sequential(nn.Linear(self.hidden_dim2, self.output_dim), nn.Softmax(dim = 1))
        
        
       
    def forward(self, x):
        self.x = self.conf(x)
        self.x2 = self.cluster(x)
        self.attn_output = self.active(self.attn_output)
        outs = self.attn_output
        self.outputs = self.fc(outs)
        
        return self.outputs
  