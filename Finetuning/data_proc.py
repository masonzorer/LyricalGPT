# data loader for finetuning
import pandas as pd
import torch
import tiktoken

# dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.enc = tiktoken.get_encoding("gpt2")
        self.title_token = [50301]
        self.genre_token = [50302]
        self.lyrics_token = [50303]
        self.end_token = [50304]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get song at index idx
        song = self.data.iloc[idx]

        # get title, genre, and lyrics
        title = song['title']
        genre = song['tag']
        lyrics = song['lyrics']

        # tokenize title, genre, and lyrics and combine into one tokenized song
        tokenized_title = self.enc.encode(title)
        tokenized_genre = self.enc.encode(genre)
        tokenized_lyrics = self.enc.encode(lyrics)
        tokenized_song = self.title_token + tokenized_title + self.genre_token + tokenized_genre + self.lyrics_token + tokenized_lyrics

        # truncate longer songs to 1024 tokens
        if len(tokenized_song) > 1023:
            tokenized_song = tokenized_song[:1023]
        tokenized_song = tokenized_song + self.end_token

        # attention mask for transformer
        mask = torch.zeros(1024)
        mask[:len(tokenized_song)] = 1

        # if token is less than 1024 tokens, pad with 0s
        if len(tokenized_song) < 1024:
            tokenized_song = tokenized_song + [0] * (1024 - len(tokenized_song))

        # create x and y
        x = torch.tensor(tokenized_song[:-1])
        y = torch.tensor(tokenized_song[1:])

        # return x and y
        return x, y, mask

def main():
    # load data
    print('Loading data...')
    data = pd.read_csv('./Data/df_eng.csv')
    print('Data loaded!')

    # create dataset and dataloader
    dataset = Dataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # get a batch of data
    x, y, mask = next(iter(dataloader))
    print('x and y shape:')
    print(x)
    print(y)
    print(mask)

    # configure tokenizer
    gpt2 = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="gpt2",
        pat_str=gpt2._pat_str,
        mergeable_ranks=gpt2._mergeable_ranks,
        special_tokens={
            **gpt2._special_tokens,
            "<|song_title|>": 50301,
            "<|song_genre|>": 50302,
            "<|song_lyrics|>": 50303,
            "<|song_end|>": 50304
        }
    )

if __name__ == '__main__':
    main()