import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import copy
from Model import LayerNorm, Block




class Client1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_outputs = {}
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])


    def get_batch_gradients(self, batch_id):
        return self.batch_outputs[batch_id].grad


    def embed_tokens(self, input_ids, past_length):
        position_ids = torch.arange(past_length, input_ids.size(-1)+past_length,
                                     dtype=torch.long, device=input_ids.device) #sequence from past to past+N
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #expand for 1 x N and match input shape B x N
        inputs_embeds = self.wte(input_ids) #embeddem' good [B,N]->[B,N,D]
        position_embeds = self.wpe(position_ids) #[B,N]->[B,N,D]
        hidden_states = inputs_embeds + position_embeds #elem wise add embeds
        return hidden_states


    def forward(self, input_ids, past=None, batch_id=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks) #init past per block
        else: 
            past_length = past[0][0].size(-1)

        hidden_states = self.embed_tokens(input_ids, past_length)
    
        presents = []
        for block, layer_past in zip(self.blocks, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        if batch_id is not None:
            hidden_states.retain_grad()
            self.batch_outputs[batch_id] = hidden_states

        return hidden_states, presents




class Client2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_outputs = {}
        block = Block(config)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
    

    def get_batch_gradients(self, batch_id):
        return self.batch_outputs[batch_id].grad


    def forward(self, hidden_states, past=None, batch_id=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks) #init past per block
        else: 
            past_length = past[0][0].size(-1)

        presents = []
        for block, layer_past in zip(self.blocks, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        if batch_id is not None:
            hidden_states.retain_grad()
            self.batch_outputs[batch_id] = hidden_states
            
        return hidden_states, presents




class Client3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_outputs = {}
        block = Block(config)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
    

    def get_batch_gradients(self, batch_id):
        grad = self.batch_outputs[batch_id].grad
        del self.batch_outputs[batch_id]
        return grad
    

    def forward(self, hidden_states, past=None, batch_id=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks) #init past per block
        else: 
            past_length = past[0][0].size(-1)

        presents = []
        for block, layer_past in zip(self.blocks, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        if batch_id is not None:
            hidden_states.retain_grad()
            self.batch_outputs[batch_id] = hidden_states

        return hidden_states, presents




class Client4(nn.Module):
    def __init__(self, config, embedding_weight):
        super().__init__()
        self.batch_outputs = {}
        block = Block(config)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size)
        self.output_head.weight = embedding_weight ### TODO find strucutre / how to load


    def get_batch_gradients(self, batch_id):
        return self.batch_outputs[batch_id].grad #grad wont populate till backward pass


    def forward(self, hidden_states, past=None, batch_id=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks) #init past per block
        else: 
            past_length = past[0][0].size(-1)
        
        presents = []
        for block, layer_past in zip(self.blocks, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)

        if batch_id is not None:
            hidden_states.retain_grad()
            self.batch_outputs[batch_id] = hidden_states
            
        return hidden_states, presents