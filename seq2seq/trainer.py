import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from model import Encoder, Decoder, Seq2Seq
from dataloader import train_dataset, val_dataset, collate_fn, vocab_transform
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# Training Hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 32

# Model Hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(vocab_transform['de'])
input_size_decoder = len(vocab_transform['en'])
output_size = len(vocab_transform['en'])
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard to get nice loss plot
writer = SummaryWriter(f'runs/loss_plot')
step = 0

# Load Data and initialize the model
train_iterator = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_iterator = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, output_size, dec_dropout).to(
                        device)
seqModel = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(seqModel.parameters(), lr=learning_rate)

pad_idx = vocab_transform['en'].get_stoi()['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# print('pair_idx')
# print(pad_idx)
# print('en vocab len', len(vocab_transform['en']))
# print('de vocab len', len(vocab_transform['de']))
#

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), seqModel, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch [{epoch}/{num_epochs}]')
    checkpoint = {'state_dict': seqModel.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx, data in enumerate(train_iterator):
        src, trg = data
        src_data = src.to(device)
        trg_data = trg.to(device)
        output = seqModel(src_data, trg_data)
        output = output[1:].reshape(-1, output.shape[2])
        target = trg_data[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seqModel.parameters(), max_norm=1)
        optimizer.step()
        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1

