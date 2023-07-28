# generate lyrics with GPT-2 model
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate():
    # load model from saved checkpoint (gpt_tuned.pth)
    print('Loading model...')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('gpt_tuned.pth'))
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
    tokenized_song = tokenizer.encode_plus(
        model_input,
        return_tensors='pt'
    )['input_ids'].flatten()

    attention_mask = torch.ones_like(tokenized_song)  # Set all elements to 1
    attention_mask[tokenized_song == tokenizer.pad_token_id] = 0

    # generate lyrics
    print('Generating lyrics...')
    generated = model.generate(
        tokenized_song.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=25,
        max_length=1024,
        num_return_sequences=1
    )
    # decode generated lyrics
    generated = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(generated)


if __name__ == '__main__':
    generate()