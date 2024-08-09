import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import copy



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

    def forward(self, X, past=None):
        X = self.W1(X)
        Q,K,V = X.split(self.d,dim=2)
        q = self.reshape(Q)
        k = self.reshape(K,k=True)
        v = self.reshape(V)
        if past is not None: 
            past_k, past_v = past[0].transpose(-2,-1), past[1] # transpose k bc computes transposed and stored untransposed 
            v = torch.cat(past[0],v)
            k = torch.cat(past[1],k)
        present = torch.stack((k.tanspose(-2,-1),v))
        out = self.heads(q,k,v)
        out = self.cat(out)
        return self.W2(out) , present


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
    

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.FFN = FFN(config,config.n_exp)
        self.ATTN = Attn(config)
        self.ln_1 = LayerNorm(config.n_emb)
        self.ln_2 = LayerNorm(config.n_emb)
    def forward(self,x):
        a = self.ATTN(x)
        x = self.ln_1(a+x) 
        m = self.FFN(x)
        x = self.ln_2(m+x)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # learnable weight matrix that can contain all embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # weights like x
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents    
    

class Head():
    def __inti__(self, emb_weights, config):
        super(Head, self).__inti__()
        self.n_emb = config.n_emb
        self.set_embedding_weights(emb_weights)
    def set_embedding_weights(self,emb_weights):
        shape = emb_weights.shape
        self.embed = nn.Linear(shape[1],shape[0], bias=False)
        self.embed.weight = emb_weights
    def forward(self, hidden):
        return self.embed(hidden)


class Heads_Model():
    def __init__(self,config):
        super(Heads_Model,self).__init__()
        self.model = Model(config)
        token_weights = self.model.wte.weight ### TODO get how this deffed / passed in 
        self.head = Head(token_weights, config)

    def set_tied(self): ### TODO what tied weights do
        self.head.set_emb_weights(self.model.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type = None, labels = None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type, past)
        lm_logits = self.lm_head(hidden_states)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss
        return lm_logits, presents
