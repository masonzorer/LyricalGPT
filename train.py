# Train a decoder model
import model
import torch
import tiktoken
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Training Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
learning_rate = 3e-4
eval_interval = 100
eval_samples = 50

# Transformer decoder model components
@dataclass
class Config:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 32
    n_embd: int = 256
    dropout: float = 0.0

# dataloader for loading a batch from .bin files
def dataloader(config, data, batch_size):
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

# estimate loss
@torch.no_grad()
def estimate_loss(model, data, config):
    model.eval()
    losses = torch.zeros(eval_samples)
    for i in range(eval_samples):
        x, y = dataloader(config, data, 2)
        logits = model(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean()

# initialize model and data
train = np.memmap('Data/train.bin', dtype=np.uint16, mode='r')
val = np.memmap('Data/val.bin', dtype=np.uint16, mode='r')
config = Config()
decoder = model.Decoder(config).to(device)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

# print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in decoder.parameters())}")

# training loop
for i in range(1500):
    # get a batch & forward pass
    x, y = dataloader(config, train, batch_size)
    logits = decoder(x)

    # compute loss
    logits = logits.view(-1, logits.size(-1))
    y = y.view(-1)
    loss = F.cross_entropy(logits, y)
    print(loss.item())

    # backward pass and update
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # evaluate
    if (i+1) % eval_interval == 0:
        train_loss = estimate_loss(decoder, train, config)
        val_loss = estimate_loss(decoder, val, config)
        print(f"Step {i+1}, Train loss: {train_loss}, Val loss: {val_loss}")

# generate a sample
x = torch.zeros((1, 1), dtype=torch.long)
sample = decoder.generate_sample(x, 100)

# decode the sample
enc = tiktoken.get_encoding("gpt2")
print(enc.decode(sample[0].tolist()))

# save the model
torch.save(decoder.state_dict(), "decoder.pth")







