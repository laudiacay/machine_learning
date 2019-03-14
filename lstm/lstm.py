import torch
import torch.nn as nn
from read_data import EMBEDDING_SIZE, PAD_TOK, START_TOK, END_TOK
import random

class LSTM(nn.Module):
    def __init__(self, inp_size, hid_size):
        super(LSTM, self).__init__()
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.Wi = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        self.Wf = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        self.Wo = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        self.Wg = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        # PERFORMANCE: add log softmax layer?

    def forward(self, x_seq, h_0, c_0):
        if h_0 is None: h_0 = self.init_hidden()
        if c_0 is None: c_0 = self.init_hidden()

        def one_step(x, h_i, c_i):
            stacked = torch.cat((x, h_i), 0)
            i = self.Wi(stacked).sigmoid()
            f = self.Wf(stacked).sigmoid()
            o = self.Wo(stacked).sigmoid()
            g = self.Wg(stacked).tanh()
            c_i1 = f * c_i + i * g
            h_i1 = o * c_i1.tanh()
            return h_i1, c_i1

        out = []
        h_t, c_t = h_0, c_0
        for x in x_seq:
            h_t, c_t = one_step(x, h_t, c_t)
            out.append(h_t)

        return out, (h_t, c_t)

    def init_hidden(self):
        return torch.zeros(self.hid_size)

class Seq2Seq(nn.Module):
    def __init__(self, hid_size, SENTENCES):
        super(Seq2Seq, self).__init__()
        self.hid_size = hid_size
        self.SENTENCES = SENTENCES
        self.vocab_size = len(self.SENTENCES.vocab)
        
        self.one_hot_emb = nn.Embedding(self.vocab_size, self.vocab_size)
        self.one_hot_emb.weight.data = torch.eye(self.vocab_size)
        self.one_hot_emb.weight.requires_grad = False

        self.embed_in = nn.Linear(self.vocab_size, EMBEDDING_SIZE)
        self.encoder = nn.LSTM(EMBEDDING_SIZE, self.hid_size)
        self.decoder = nn.LSTM(EMBEDDING_SIZE, self.hid_size)
        self.embed_out = nn.Linear(self.hid_size, self.vocab_size)
    
    def init_hidden(self, batch_size):
        return torch.zeros([batch_size, self.hid_size]).unsqueeze(0)

    def forward(self, first_sent, target, teacher_forcing_rate=0.5):
        batch_size = first_sent.shape[1]
        embedded_input = self.embed_in.forward(self.one_hot_emb(first_sent))
        h_i, c_i = self.init_hidden(batch_size), self.init_hidden(batch_size)
        _, (h_i, c_i) = self.encoder.forward(embedded_input, (h_i, c_i))

        outputs = []

        if target is not None:
            embedded_trget = self.embed_in.forward(self.one_hot_emb(target))
            x_i = embedded_trget[0, :].unsqueeze(0)
            for i in range(1, embedded_trget.shape[0]):
                dec_out, (h_i, c_i) = self.decoder.forward(x_i, (h_i, c_i))
                output = self.embed_out(dec_out)
                outputs.append(output)
                if random.random() > teacher_forcing_rate:
                    pred = output.max(2)[1]
                    x_i = self.embed_in(self.one_hot_emb(pred))
                else:
                    x_i = embedded_trget[i].unsqueeze(0)
        else: # test mode, not batch mode
            x_i = torch.tensor([[self.SENTENCES.vocab.stoi[END_TOK]]])
            x_i = self.embed_in(self.one_hot_emb(x_i.long()))
            while True:
                dec_out, (h_i, c_i) = self.decoder.forward(x_i, (h_i, c_i))
                output = self.embed_out(dec_out)
                outputs.append(output)
                pred = output.max(2)[1].squeeze(0)
                x_i = self.embed_in(self.one_hot_emb(pred)).unsqueeze(0)
                if pred[0] == self.SENTENCES.vocab.stoi[END_TOK]:
                    break
        outputs = torch.cat(outputs).permute([0, 2, 1])
        
        return outputs
