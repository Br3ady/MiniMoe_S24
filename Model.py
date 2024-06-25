import torch
import torch.nn as nn
import torch.nn.functional as F 
import math



def gelu(x):
    theta_x = (1+torch.nn.tanh(math.sqrt(2/math.pi) * (x+0.044715*torch.pow(x,3))))
    return 0.5 * x * theta_x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-12):
        super(LayerNorm,self).__init__()
        self.Weight = nn.Parameter(torch.ones(size))
        self.Bias = nn.Parameter(torch.zeros(size))
        self.eps = eps
    def forward(self, x):
        mean = torch.mean(x)
        var = torch.var(x)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.Weight + self.Bias
        

class Attn(nn.Module):
    def __init__(self,config):
        super(Attn,self).__init__()
        d = config.n_embd
        self.W = nn.Linear(d,3*d)
    def head(self,x):
        pass
    def forward(self, x):
         Q,K,V = torch.split(self.W(x),3)

         

        

