import torch
import torch.nn as nn

class SingleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        
        self.input_dim = input_dim # 22
        self.hidden_dim = hidden_dim 
        self.hidden_dim2 = hidden_dim//4
        self.hidden_dim3 = hidden_dim//8
        self.output_dim = output_dim #3
        
        self.Layer1 = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())  #hidden layer
        self.Layer2 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim2), nn.ReLU())
        self.Layer3 = nn.Sequential(nn.Linear(self.hidden_dim2, self.hidden_dim3), nn.ReLU())
        # self.Layer4 = nn.Linear(self.hidden_dim3, self.output_dim)
        self.Layer4 = nn.Sequential(nn.Linear(self.hidden_dim3, self.output_dim), nn.Softmax(dim = 1))
        
        
       
    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
       
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



class FocalLoss_ver2(nn.Module):
  def __init__(self, gamma=0, alpha=None, size_average=True):
    super(FocalLoss_ver2, self).__init__()
    self.gamma = gamma
    self.alpha = alpha
    #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
    #if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
    self.size_average = size_average
 
  def forward(self, input, target, weights=None):
      if input.dim()>2:
          input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
          input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
          input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
      #target = target.view(-1,1)
      target = target.type(torch.int64)
    
      logpt = F.log_softmax(input)
    #   print(f'F.log_softmax(input) |{logpt}') 
      logpt = logpt.gather(1,target)
    #   print(f'logpt.gather(1,target) |{logpt}') 
      logpt = logpt.view(-1)
    #   print(f'logpt.view(-1) |{logpt}') 
      pt = Variable(logpt.data.exp())
    #   print(f'pt = Variable(logpt.data.exp()) |{logpt}') 
      
    #   if self.alpha is not None:
    #       if self.alpha.type()!=input.data.type():
    #           self.alpha = self.alpha.type_as(input.data)
    #       at = self.alpha.gather(0,target.data.view(-1))
    #       logpt = logpt * Variable(at)
      #logpt = logpt * 2
    #   loss = -1 * (1-pt)**self.gamma * logpt
      loss = -self.alpha * (1-pt)**self.gamma * logpt
      if self.size_average: return loss.mean()
      else: 
        # if not weights == None:
        #     loss = loss*weights
        # else:
        #     loss= loss
              
        return loss.sum()
      
    