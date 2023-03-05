import sys
import torch
import spacy
from torchtext.data.metrics import bleu_score
from torchtext.data.utils import get_tokenizer


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Load german tokenizer
    # Define the tokenizers
    token_transform = {'de': get_tokenizer('spacy', language='de_core_news_sm'),
                       'en': get_tokenizer('spacy', language='en_core_web_sm')}
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in token_transform['de'](sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<bos>')
    tokens.append('<eos>')
    # pad_idx = vocab_transform['en'].get_stoi()['<pad>']

    # Go through each german token and convert to an index
    text_to_indices = [german.get_stoi()[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.get_stoi()["<bos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.get_stoi()["<eos>"]:
            break

    translated_sentence = [english.get_itos()[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    try:
        print("===> Saving checkpoint ")
        torch.save(state, filename)
        print("===> Checkpoint saved ")
    except ResourceWarning as e:
        print("===> Error Checkpoint could not be saved ")


def load_checkpoint(checkpoint, model, optimizer):
    try:
        print("==> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("==> Loaded checkpoint successfully")
    except ResourceWarning as e:
        print("==> Error No checkpoint was loaded")
