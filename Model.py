import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import copy


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
        self.n_ctx = config.n_ctx
        self.d = config.n_embd #embed dim
        self.h = config.n_head #number of attn heads (div along d)
        assert self.d % self.h == 0 #make sure div equally
        self.W1 = nn.Linear(self.d,3*self.d) #for expanding input [B,N,D] into q,k,v like x
        self.out_proj = nn.Linear(self.d,self.d) ### more learnable params, no shape change

    def reshape(self, x, k=False):
        in_shape = x.size()[:-1] + (self.h, self.d // self.h) #[B,N,D] -> [B,N,H,D/H] i.e div emb into heads
        x = x.view(in_shape)
        if k:
            x = x.permute(0,2,3,1) #[B,N,H,D/H] -> [B,H,D/H,N] 
        else:
            x = x.permute(0,2,1,3) #[B,N,H,D/H] -> [B,H,N,D/H] #mult per head along seq for scores 
        return x
    
    def cat(self, x):
        out_shape = (x.size()[0],) + (x.size()[2],) + (x.size()[3]*x.size()[1],) #[B,H,D/H,N] -> [B,N,D] return to shape and cat last 2 dims
        x = x.view(out_shape)
        return x

    def heads(self, q,k,v, mask=None):
        w = (torch.matmul(q,k)) / math.sqrt(q.size(-1)) #ATTN calc 
        if mask is not None:
            w = w.masked_fill(mask==0,float('-inf')) #upper right mask 
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w,v)

    def forward(self, X, layer_past): #past[i] for block i
        X = self.W1(X) 
        Q,K,V = X.split(self.d,dim=2) #split to q,k,v
        q = self.reshape(Q) # to mult shapes
        k = self.reshape(K,k=True)
        v = self.reshape(V)
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-1) # dim -1 since k is always transposed so N is last
            v = torch.cat([past_v, v], dim=-2) # cat over N
            if k.size(-2) > self.n_ctx: # truncate at context length
                k = k[:, :, :, -self.n_ctx:]
                v = v[:, :, -self.n_ctx:, :]

        layer_present = (k,v) #save updated k,v (wont be used during training)
        _,_,N,_ = q.size()
        mask = torch.tril(torch.ones((N,N), device=q.device)).view(1,1,N,N) ### TODO why shape
        out1 = self.heads(q,k,v, mask=mask) #do ATTN
        out2 = self.cat(out1) #combine heads
        return self.out_proj(out2), layer_present 


class FFN(nn.Module):
    def __init__(self, config): 
        super(FFN, self).__init__()
        d = config.n_embd
        hidden = d*4 #good ratio (scale more for wider model)
        self.ln_1 = nn.Linear(d, hidden)
        self.c_proj = nn.Linear(hidden, d)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.ln_1(x)) #A.E for compexity learning
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.ATTN = Attn(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.FFN = FFN(config)

    def forward(self, x, layer_past):
        a, layer_present = self.ATTN(x,layer_past) #past -> present
        x = self.ln_1(a+x) 
        m = self.FFN(x)
        x = self.ln_2(m+x)
        return x, layer_present #pass out for appending to total past



##### NOT USED FOR SPLIT TRAINING ##### 



class Model(nn.Module):  
    def __init__(self, config):
        super(Model, self).__init__()
        self.n_layer = config.n_layer 
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size ###ensure match w/ tokenizer

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # learnable lookup tabel vocab -> embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # ^^ but for position
        block = Block(config)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)]) # new copy of blocks
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, past=None): #take in [B,N] token ids (past = presents from infer loop)
        if past is None: #always true for training
            past_length = 0
            past = [None] * len(self.h) #needed for safe append
        else:
            past_length = past[0][0].size(-1) #current N dim of saved k

        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                    device=input_ids.device) #sequence for length N (if past shift by past_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #expand for 0 x N and match input shape B x N

        input_shape = input_ids.size() #[B,N]
        input_ids = input_ids.view(-1, input_ids.size(-1)) #does nothing rn bc shape 2d already (used for non-scuffed microbatching), mine is scuffed tho
        position_ids = position_ids.view(-1, position_ids.size(-1)) #^^

        inputs_embeds = self.wte(input_ids) #embeddem' good
        position_embeds = self.wpe(position_ids)

        
        hidden_states = inputs_embeds + position_embeds #elem wise add embeds
        presents = [] #reset for new presents

        for block, layer_past in zip(self.h, past): 
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present) #one present per block
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),) #reshape out to [B,N,D]
        return hidden_states.view(*output_shape), presents


class Head(nn.Module): ## decode ffn output to logits over vocab 
    def __init__(self, model_embeddings_weights, config):
        super(Head, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False) # similarity measure between output sequence and token embeddings 
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
        return lm_logits, presents #passed in as past in next forward pass (store in inference loop useless in training)

