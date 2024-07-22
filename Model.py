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
    def __init__(self, config):
        super(Attn,self).__init__()
        self.d = config.n_embd
        self.h = config.n_head
        assert self.d % self.h == 0
        self.W1 = nn.Linear(self.d,3*self.d)
        self.W2 = nn.Linear(self.d,self.d)

    def reshape(self, x, k=False):
        in_shape = x.size()[:-1] + (self.h, self.d // self.h)
        x = x.view(in_shape)
        if k:
            x = x.permute(0,2,1,3)
        else:
            x = x.permute(0,2,3,1)
        return x
    
    def cat(self, x):
        out_shape = (x.size()[0],) + (x.size()[3],) + (x.size()[2]*x.size()[1],)
        x = x.view(out_shape)
        return x

    def heads(self, q,k,v):
        w = (torch.matmul(q,k)) / math.sqrt(q.size(-1))
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w,v)

    def forward(self, X):
        X = self.W1(X)
        Q,K,V = X.split(self.d,dim=2)
        q = self.reshape(Q)
        k = self.reshape(K,k=True)
        v = self.reshape(V)
        out = self.heads(q,k,v)
        out = self.cat(out)
        return self.W2(out)


class FFN(nn.Module):
    def __init__(self, config, n_state):  # in MLP: n_state=3072 (4 * n_embd)
        super(FFN, self).__init__()
        nx = config.n_embd
        self.c_fc = nn.Linear(nx, n_state)
        self.c_proj = nn.Linear(n_state, nx)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class TopKRoute(nn.Module):
    def __init__(self, nx, n_exp, k):
        super(TopKRoute,self).__init__()
        self.W = nn.Linear(nx, n_exp)
        self.Softmax = nn.Softmax(dim=1)
        self.k = k

    def forward(self,x):
        scores = self.W(x)
        scores = scores.mean(dim=1)
        scores = self.Softmax(scores)
        vals,idx = torch.topk(scores, self.k, dim=-1)
        return vals, idx
    

class FFN_Experts(nn.Module):
    def __init__(self, config, n_exp):
        super(FFN_Experts,self).__init__()
        self.n_exp = n_exp
        nx = config.n_embd
        self.experts = nn.ModuleList([FFN(config,config.n_embd*4) for _ in range(n_exp)])
        self.route = TopKRoute(nx, n_exp, config.k)

    def forward(self,x):
        B,N,D = x.size()
        vals,idx = self.route(x)
        idx = idx.unsqueeze(1).expand(-1,N,-1) # reshape to [B,N,k] for matmul w/ expert output
        ### TODO Data parallel 
        outs = torch.stack([expert(x) for expert in self.experts],dim=1)
        k_outs = torch.gather(outs, 1, idx.unsqueeze(-1).expand(-1, -1, -1, outs.size(-1)))  # Shape: (B, N, k, expert_dim)
        k_vals = vals.unsqueeze(1).unsqueeze(-1).expand(-1,N,-1,k_outs.size(-1))
        out = (k_vals * k_outs).sum(dim=2)
        return out
        


class Block(nn.Module):
    def __inti__(self, config):
        super(Block, self).__init__()
        self.EXP = FFN_Experts(config,config.n_exp)
        self.ATTN = Attn(config)
        self.ln_1 = LayerNorm(config.n_emb)
        self.ln_2 = LayerNorm(config.n_emb)
    def forward(self,x):
        a = self.ATTN(x)
        x = self.ln_1(a+x) ###check add norm layer
        m = self.EXP(x)
        x = self.ln_2(m+x)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()