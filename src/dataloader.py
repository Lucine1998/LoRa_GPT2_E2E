import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer

def pad_tokens(tokens, max_seq_length, padding_token):
    res_tokens = tokens[:max_seq_length]
    token_len = len(res_tokens)
    res_tokens = res_tokens + [padding_token for _ in range(max_seq_length - token_len)]
    return res_tokens

def fill_ignore_label(labels, context_tokens):
    labels[:len(context_tokens) - 1] = [-100] * (len(context_tokens) - 1)
    return labels

def collate_batch(batch, tokenizer, block_size, device):
    context_list = [c + "\n" for c in list(zip(*batch))[0]]
    completion_list = list(zip(*batch))[1]

    context_result = tokenizer(context_list, padding=True, truncation=True, max_length=block_size, return_tensors='pt')
    completion_result = tokenizer(completion_list, padding=True, truncation=True, max_length=block_size, return_tensors='pt')

    inputs = torch.cat((context_result["input_ids"], completion_result["input_ids"]), dim=1)
    masks = torch.cat((context_result["attention_mask"], completion_result["attention_mask"]), dim=1)

    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    labels = torch.cat((context_result["input_ids"], completion_result["input_ids"]), dim=1)[:, 1:]
    labels = torch.cat((labels, torch.tensor([[eos_id] * len(labels)], dtype=torch.long)), dim=1)
    labels = list(map(lambda l, c: fill_ignore_label(l.tolist(), c.tolist()), labels, context_result["input_ids"]))
    labels = torch.tensor(labels, dtype=torch.long)

    inputs = pad_tokens(inputs.tolist(), block_size, 0)
    masks = pad_tokens(masks.tolist(), block_size, 0)
    labels = pad_tokens(labels.tolist(), block_size, -100)

    return torch.tensor(inputs).to(device), torch.tensor(labels).to(device), torch.tensor(masks).to(device)

def get_dataloader(data_path, tokenizer, batch_size, block_size, device, shuffle=True):
    data = pd.read_json(data_path, lines=True)
    dataset = list(zip(data["context"], data["completion"]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_batch(x, tokenizer, block_size, device))
