# generate lyrics with GPT-2 model
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@torch.no_grad()
def generate():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model from saved checkpoint (gpt_tuned.pth)
    print('Loading model...')
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(torch.load("gpt_tuned.pth", map_location=device))
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # get model input
    song_title = input('Enter song title: ')
    song_genre = input('Enter song genre: ')
    song_lyrics = input('Enter song lyrics: ')
    model_input = song_title + '\n' + song_genre + '\n\n' + song_lyrics

    # tokenize model input
    output = tokenizer.encode_plus(
        model_input,
        return_tensors='pt'
    )

    tokenized_song = output['input_ids']
    tokenized_song = tokenized_song.to(device)
    
    # generate lyrics
    print('Generating lyrics...')
    print()
    print(model_input, end='', flush=True)
    for i in range(900):
        output = model(tokenized_song)
        next_token_logits = output[0][:, -1, :]
        # sample next token from distribution and append to generated lyrics
        scores = torch.softmax(next_token_logits, dim=1)
        next_token = torch.multinomial(scores, num_samples=1)
        # update model input
        tokenized_song = torch.cat((tokenized_song, next_token), dim=1)
        # stop if end of song token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
        # print generated lyrics to buffer immediately
        decoded_token = tokenizer.decode(next_token.item())
        print(decoded_token, end='', flush=True)
    print()


if __name__ == '__main__':
    generate()