# Transformer decoder model components
import torch
import torch.nn as nn
import torch.nn.functional as F

# one head of self-attention
class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.n_embd // config.n_head
        # query, key, value projections
        self.q = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.k = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.v = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # calculate query, key, values
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # compute attention scores
        att = q @ k.transpose(-2, -1) * C ** (-0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # attend to values
        y = att @ v
        return y
    
# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# feed forward network
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        return self.net(x)
    
# full transformer decoder block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffn = FFN(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# main model class
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # create token and postion embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        B, T = x.shape
        # embed tokens and positions
        tok_embeddings = self.token_embed(x)
        pos_embeddings = self.pos_embed(torch.arange(T, device=x.device))
        x = tok_embeddings + pos_embeddings
        # pass through transformer blocks
        x = self.blocks(x)
        # project back to vocabulary
        x = self.head(self.ln(x))
        return x
    
    def generate_sample(self, x, length):
        # input x is a single token of shape (B, 1)
        # generate a sequence of length `length`
        for _ in range(length):
            # crop context to block size 
            x = x[:, -self.block_size:]
            # get predictions for the next token
            logits = self(x)
            logits = logits[:, -1, :]
            # get probabilities and sample
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, 1)
            # append the new token to the sequence
            x = torch.cat((x, x_next), dim=1)
        return x