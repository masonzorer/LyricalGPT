# data loader for finetuning
import pandas as pd
import torch
import tiktoken
from transformers import GPT2Tokenizer

# dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, GPT=False):
        self.GPT = GPT
        self.data = data
        self.enc = tiktoken.get_encoding("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.title_token = [50300]
        self.genre_token = [50301]
        self.lyrics_token = [50302]
        self.end_token = [50303]

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
        if self.GPT:
            text = title + '\n' + genre + '\n\n' + lyrics
            tokenized_song = self.tokenizer.encode_plus(
                text,
                max_length=1024,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # create x and attention mask
            x = tokenized_song['input_ids'].flatten()
            attention_mask = tokenized_song['attention_mask'].flatten()
            return x, attention_mask # y is the same as x
        else:
            tokenized_title = self.enc.encode(title)
            tokenized_genre = self.enc.encode(genre)
            tokenized_lyrics = self.enc.encode(lyrics)
            tokenized_song = self.title_token + tokenized_title + self.genre_token + tokenized_genre + self.lyrics_token + tokenized_lyrics
            # truncate longer songs to 1024 tokens
            if len(tokenized_song) > 1023:
                tokenized_song = tokenized_song[:1023]
            tokenized_song = tokenized_song + self.end_token

            # if token is less than 1024 tokens, pad with 0s
            if len(tokenized_song) < 1024:
                tokenized_song = tokenized_song + [0] * (1024 - len(tokenized_song))

            # create x and y
            x = torch.tensor(tokenized_song[:-1])
            y = torch.tensor(tokenized_song[1:])

            # return x and y
            return x, y

def main():
    # load data
    print('Loading data...')
    data = pd.read_csv('./Data/train.csv')
    print('Data loaded!')

    # create dataset and dataloader
    dataset = Dataset(data, GPT=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # get a batch of data
    x, y = next(iter(dataloader))
    print('x and y shape:')
    print(x)
    print(y)

if __name__ == '__main__':
    main()