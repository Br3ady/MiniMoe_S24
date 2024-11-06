import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from Model_config import Config
import Model


class OpenWebText(Dataset):
    def __init__(self, config, tokenizer, token_set):
        super(OpenWebText,self).__init__()
        self.data = token_set
        self.tokenizer = tokenizer
        self.seq_len = config.n_ctx

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the tokenized input_ids
        input_ids = self.data["input_ids"][idx]

        # Truncate or pad the input_ids to the desired sequence length
        input_ids = input_ids[:self.seq_len]
        if len(input_ids) < self.seq_len:
            input_ids += [self.tokenizer.pad_token_id] * (self.seq_len - len(input_ids))

        # Input is the token sequence, target is the shifted sequence
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = input_ids.clone()
        target_ids[:-1] = input_ids[1:]  # Shifted target

        return {"input_ids": input_ids, "labels": target_ids}

