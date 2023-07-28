# fine tune GPT2 on the song lyrics dataset
import data_proc
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# Training Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 20
batch_size = 3
learning_rate = 1e-5
eval_interval = 200
eval_samples = 50

# estimate loss
@torch.no_grad()
def estimate_loss(model, data_loader):
    model.eval()
    losses = torch.zeros(eval_samples)
    for i in range(eval_samples):
        # grab a batch of data
        x, mask = next(iter(data_loader))
        x = x.to(device)
        mask = mask.to(device)
        # forward pass
        outputs = model(x, attention_mask=mask, labels=x)
        loss = outputs.loss
        losses[i] = loss.item()
    model.train()
    return losses.mean()

# initialize data
train_set = pd.read_csv('./Data/train.csv')
val_set = pd.read_csv('./Data/val.csv')
train_dataset = data_proc.Dataset(train_set, GPT=True)
val_dataset = data_proc.Dataset(val_set, GPT=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# initizlize model and optimizer
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc., for larger versions
gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
gpt_model.to(device)
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)

# print number of parameters
num_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)

# training loop
for epoch in range(epochs):
    for i, (x, mask) in enumerate(train_loader):
        # get batch of data
        x = x.to(device)
        mask = mask.to(device)
        # forward pass
        output = gpt_model(x, attention_mask=mask, labels=x)
        loss = output.loss

        print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss
        if (i+1) % eval_interval == 0:
            train_loss = estimate_loss(gpt_model, train_loader)
            val_loss = estimate_loss(gpt_model, val_loader)
            print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
            torch.save(gpt_model.state_dict(), "gpt_tuned.pth")