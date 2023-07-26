# Generate lyrics using the trained model
import model
import torch
import tiktoken
from dataclasses import dataclass

# Transformer decoder model components
@dataclass
class Config:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

def generate():
    # load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    decoder = model.Decoder(config).to(device)
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
    decoder.eval()
    # initialize the tokenizer
    gpt2 = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="gpt2",
        pat_str=gpt2._pat_str,
        mergeable_ranks=gpt2._mergeable_ranks,
        special_tokens={
            **gpt2._special_tokens,
            "<|song_title|>": 50300,
            "<|song_genre|>": 50301,
            "<|song_lyrics|>": 50302,
            "<|song_end|>": 50303
        }
    )
    # get input from user
    title = input("Enter song title: ")
    genre = input("Enter song genre: ")
    tokenized_title = enc.encode(title)
    tokenized_genre = enc.encode(genre)
    model_input = torch.tensor([50300] + tokenized_title + [50301] + tokenized_genre + [50302]).unsqueeze(0).to(device)
    
    # generate lyrics
    for _ in range(128):
        logits = decoder(model_input)
        logits = logits[:, -1, :]
        # sample from the logits
        next_token = torch.multinomial(logits.softmax(-1), num_samples=1)
        model_input = torch.cat((model_input, next_token), dim=1)
        if next_token == 50303:
            break
    # decode the generated lyrics
    generated_lyrics = enc.decode(model_input[0].tolist())
    generated_lyrics = generated_lyrics.replace("<|song_title|>", "")
    generated_lyrics = generated_lyrics.replace("<|song_genre|>", "")
    generated_lyrics = generated_lyrics.replace("<|song_lyrics|>", "")
    generated_lyrics = generated_lyrics.replace("<|song_end|>", "")
    print(generated_lyrics)


if __name__ == "__main__":
    generate()