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


class Model(nn.Module): ## GPT architecture 
    def __init__(self, config):
        super(Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # lookup tabel vocab -> learnable_embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # ^^ 
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)]) # new copy of blocks
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, past):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        token_type_embeds = 0
        
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []

        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class Head(nn.Module): ## head to decode ffn output to logits over vocab 
    def __init__(self, model_embeddings_weights, config):
        super(Head, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False) # essentially similarity measure between output sequence and token embeddings 
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT(nn.Module): ## Model + Head
    def __init__(self, config):
        super(GPT, self).__init__()
        self.transformer = Model(config)
        self.lm_head = Head(self.transformer.wte.weight, config)

    def set_tied(self):
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, past, lm_labels=None):
        hidden_states, presents = self.transformer(input_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None: # Training
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents
    


    