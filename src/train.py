import torch
from transformers import AdamW
from tqdm import tqdm

def train(model, dataloader, optimizer, device, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, labels, masks in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

def save_model(model, path="finetuned_model"):
    model.save_pretrained(path)
