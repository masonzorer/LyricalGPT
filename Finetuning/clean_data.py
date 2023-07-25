# clean and preprocess song lyric data for finetuning
import pandas as pd
import tiktoken

def load_songs():
    ' load songs and save relevant columns '
    chunk_size = 100000
    selected_columns = ['title', 'tag', 'lyrics']
    df_chunks = pd.read_csv('Data/ds2.csv', chunksize=chunk_size, usecols=selected_columns)
    df = pd.concat(df_chunks, ignore_index=True)

    # remove songs with no lyrics (including instrumental songs)
    df = df[df['lyrics'].notna()]
    df = df[df['lyrics'] != '[Instrumental]']
    print(df.shape)
    
    # save songs to csv file
    df.to_csv('Data/songs.csv', index=False)

def create_samples():
    ' create samples from songs df'
    df = pd.read_csv('Data/songs.csv')

    # tokens for seperating song pieces
    title_token = [50302]
    genre_token = [50303]
    lyrics_token = [50304]




def main():
    # each function call is a step in the data cleaning process
    load_songs()

if __name__ == '__main__':
    main()