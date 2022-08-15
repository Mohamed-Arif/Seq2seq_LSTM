# This python code has been referred from https://github.com/hoang-tn-nguyen/Disfl-QA-Performer/models.py

import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.voc_emb = nn.Embedding(vocab_size, embed_dim)
        self.size = vocab_size

    def forward(self, input):
        output = self.voc_emb(input) # (B,L,E)
        return self.dropout(output)

# --- LSTM Baseline ---
class LSTM_Enc(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = word_emb
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        
    def forward(self, input):
        input = self.embedding(input)
        outputs, (hidden, cell) = self.rnn(input)
        return hidden, cell

class LSTM_Bi_Enc(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = word_emb
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=True, batch_first=True)
        self.fc_hid = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cel = nn.Linear(hid_dim * 2, hid_dim)
        
    def forward(self, input):

        input = self.embedding(input)
        outputs, (hidden, cell) = self.rnn(input)

        hidden = hidden.view(-1, 2, hidden.shape[1], hidden.shape[2]).permute(0,2,1,3) # (N,B,D,E)
        cell = cell.view(-1, 2, cell.shape[1], cell.shape[2]).permute(0,2,1,3) # (N,B,D,E)
        hidden = hidden.reshape(hidden.shape[0], hidden.shape[1], -1) # (N,B,D*E)
        cell = cell.reshape(cell.shape[0], cell.shape[1], -1) # (N,B,D*E)
        
        hidden, cell = self.fc_hid(hidden), self.fc_cel(cell)
        return hidden, cell

class LSTM_Dec(nn.Module):
    def __init__(self, word_emb, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = word_emb
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)    
        self.prediction = nn.Sequential(
            nn.LayerNorm(emb_dim), # Without this layer, the model was very bad.
            nn.Linear(in_features=hid_dim, out_features=self.embedding.size),
            nn.Softmax(dim=-1),
        )

    def forward(self, hidden, cell, output=None, max_len=100, sid=1):

        if output == None:
            bsz = hidden.shape[1]
            output = torch.full((bsz, 1), fill_value=sid, device=hidden.device)
            
            for i in range(1,max_len):
                out_emb, (hidden, cell) = self.rnn(self.embedding(output[:,-1].unsqueeze(-1)), (hidden, cell))
                next_output = self.prediction(out_emb).max(dim=-1)[1]
                output = torch.cat([output, next_output], dim=1)
        else:
            output = self.embedding(output)
            output, (hidden, cell) = self.rnn(output, (hidden, cell))     
            output = self.prediction(output)   
        return output

class LSTM_ED(nn.Module):
    def __init__(self, src_vocab_emb, tgt_vocab_emb, emb_dim, hid_dim, n_layers, dropout=0.1, bidirectional=True):
        super().__init__()
        if bidirectional:
            self.encoder = LSTM_Bi_Enc(src_vocab_emb, emb_dim, hid_dim, n_layers, dropout)
        else:
            self.encoder = LSTM_Enc(src_vocab_emb, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = LSTM_Dec(tgt_vocab_emb, emb_dim, hid_dim, n_layers, dropout)
        
    def forward(self, input, output=None, input_len=None):
        hidden, cell = self.encoder(input)
        output = self.decoder(hidden, cell, output)
        return output


#--- Early Stopping ---    
class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


if __name__ == '__main__':
    enc = WordEmbedding(1000, 128)
    inp = torch.randint(0,1000,(8,100))
    out = torch.randint(0,1000,(8,30))
    model = LSTM_ED(enc, enc, 128, 128, 4, 0.2, bidirectional=True)
    model(inp).shape


