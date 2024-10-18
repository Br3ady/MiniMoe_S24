import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("Skylion007/openwebtext")
train_set = dataset['train']
tokenizer = AutoTokenizer.from_pretrained("gpt2") ### HF gives tokenier for model 

def tokenize(dataset):
    return tokenizer(dataset['text'], )

class OpenWebText(Dataset):
    def __init__(self, path, L):
        super(OpenWebText,self).__init__()
        self.data = dataset.load_from_disk(path)
        
        self.L = L

    def __len__(self): 
        return(self.L)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.L]
        y = self.data[idx+1: idx + 1 + self.block_size]
