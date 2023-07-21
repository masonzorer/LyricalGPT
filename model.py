# Transformer decoder model components
import torch
import torch.nn as nn
import torch.nn.functional as F


# main model class
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # create embedding matrix (Batch, Seq, Embedding)
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)

    def forward(self, x):
        # embed input tokens
        x = self.token_embed(x)

        return x
    
    def generate_sample(self, x, length):
        # input x is a single token of shape (B, 1)
        # generate a sequence of length `length`
        for _ in range(length):
            # get logits of the last token
            logits = self.forward(x)[:, -1]
            # get probabilities of the last token
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most probable
            x_next = torch.multinomial(probs, 1)
            # append the new token to the sequence
            x = torch.cat((x, x_next), dim=1)
        return x