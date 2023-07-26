# fine tune the decoder on the song lyrics dataset
import model
import data_proc
import torch
from dataclasses import dataclass
import pandas as pd
import torch.nn.functional as F

# Training Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 3
batch_size = 1
learning_rate = 1e-5
eval_interval = 50
eval_samples = 10

# Transformer decoder model components
@dataclass
class Config:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

# estimate loss
@torch.no_grad()
def estimate_loss(model, data_loader, config):
    model.eval()
    losses = torch.zeros(eval_samples)
    for i in range(eval_samples):
        # grab a batch of data
        x, y, mask = next(iter(data_loader))
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        # forward pass
        logits = model(x, mask)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean()

# initialize model and data
train_set = pd.read_csv('./Data/train.csv')
val_set = pd.read_csv('./Data/val.csv')
train_dataset = data_proc.Dataset(train_set)
val_dataset = data_proc.Dataset(val_set)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
config = Config()
decoder = model.Decoder(config).to(device)
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
decoder.train()
optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

# print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in decoder.parameters())}")

# training loop
for epoch in range(epochs):
    for i, (x, y, mask) in enumerate(train_loader):
        # get batch of data
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        # forward pass
        logits = decoder(x, mask)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss
        if (i+1) % eval_interval == 0:
            train_loss = estimate_loss(decoder, train_loader, config)
            val_loss = estimate_loss(decoder, val_loader, config)
            print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
            torch.save(decoder.state_dict(), "lyrics_gen.pth")