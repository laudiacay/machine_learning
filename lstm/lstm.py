import torch
import torch.nn as nn
from read_data import loader, EMBEDDING_SIZE, PAD_TOK, START_TOK, END_TOK

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

class EncoderLSTM(nn.Module):
    def __init__(self, hid_size, SENTENCES):
        super(EncoderLSTM, self).__init__()
        self.hid_size = hid_size

        # TODO change to your own LSTM
        self.lstm = nn.LSTM(EMBEDDING_SIZE, self.hid_size)

        # PERFORMANCE: add dropout?
        self.embed = nn.Embedding(len(SENTENCES.vocab),
                            EMBEDDING_SIZE,
                            padding_idx=SENTENCES.vocab.stoi[PAD_TOK])
        self.embed.weight.data.copy_(SENTENCES.vocab.vectors)

    def forward(self, x):
        # TODO: this line might be problematic?
        emb_x = self.embed(x)
        _, (h_t, c_t) = self.lstm(emb_x)
        # TODO: do you need to transpose/make contiguous/view h_t?
        return h_t, c_t


class DecoderLSTM(nn.Module):
    def __init__(self, hid_size, SENTENCES):
        super(DecoderLSTM, self).__init__()
        self.hid_size = hid_size
        out_size = len(SENTENCES.vocab)

        # TODO change to your own LSTM
        self.lstm = nn.LSTM(EMBEDDING_SIZE, self.hid_size)

        self.embed = nn.Embedding(out_size, EMBEDDING_SIZE,
                            padding_idx=SENTENCES.vocab.stoi[PAD_TOK])
        self.embed.weight.data.copy_(SENTENCES.vocab.vectors)

        self.out_layer = nn.Linear(self.hid_size, out_size)
        # PERFORMANCE: add dropout?

    def forward(self, x, h_t, c_t):
        # TODO: do i need to unsqueeze x?
        emb_x = self.embed(x)
        out, (h_t, c_t) = self.lstm(emb_x, (h_t, c_t))
        # TODO: do i need to squeeze out?
        pred = self.out_layer(out)
        return pred, (h_t, c_t)


class Seq2Seq(nn.Module):
    def __init__(self, hid_size, SENTENCES):
        super(Seq2Seq, self).__init__()
        self.hid_size = hid_size
        self.SENTENCES = SENTENCES

        self.encoder = EncoderLSTM(self.hid_size, self.SENTENCES)
        self.decoder = DecoderLSTM(self.hid_size, self.SENTENCES)

    def forward(self, input, target):
        outputs = []

        h_i, c_i = self.encoder.forward(input)

        if target is not None:
            x_i = target[0, :]
            # TODO I think there's maybe a type/embedding issue here?
            for i in range(1, target.shape[0]):
                output, (h_i, c_i) = self.decoder.forward(x_i, h_i, c_i)
                outputs.append(output)
                x_i = target[i]
        else:
            x_i = self.SENTENCES.stoi[START_TOK]
            while True:
                output, (h_i, c_i) = self.decoder.forward(x_i, h_i, c_i)
                outputs.append(output)
                pred = output.max(1)[1]
                x_i = pred
                if pred == self.SENTENCES.stoi[END_TOK]:
                    break

        return outputs
