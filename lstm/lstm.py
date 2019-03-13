import torch
import torch.nn as nn
import torch.nn.functional as F
import read_data

LEARNING_RATE = 1e-3

torch.manual_seed(42)

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
            print(x.shape)
            print(h_i.shape)
            stacked = torch.cat((x, h_i), 0)
            i = F.sigmoid(self.Wi(stacked))
            f = F.sigmoid(self.Wf(stacked))
            o = F.sigmoid(self.Wo(stacked))
            g = F.tanh(self.Wg(stacked))
            c_i1 = f * c_i + i * g
            h_i1 = o * F.tanh(c_i1)
            return h_i1, c_i1

        out = []
        h_t, c_t = h_0, c_0

        for x in x_seq:
            h_t, c_t = one_step(x, h_t, c_t)
            out.append(h_t)

        return out, (h_t, c_t)

    def init_hidden(self):
        return torch.zeros(1, self.hid_size)

def train(sentences, enc_model, dec_model, enc_opt, dec_opt, criterion):
    sent_1, sent_2 = sentences
    _, (hidden, ctx) = enc_model(sent_1, None, None)
    output, _ = dec_model(sent_2, hidden.detach(), ctx.detach())

    enc_opt.zero_grad()
    dec_opt.zero_grad()
    loss = 0
    for out, exp_out in zip(output, sent_2):
        loss += criterion(out, exp_out)
    loss.backward()
    enc_opt.step()
    dec_opt.step()
    return loss

train_data, dev_data, test_data = read_data.get_embedded_data()

enc_model = LSTM(200, 200)
dec_model = LSTM(200, 200)

# PERFORMANCE: torch.optim.Adam?
enc_opt = torch.optim.SGD(enc_model.parameters(), lr=LEARNING_RATE)
dec_opt = torch.optim.SGD(dec_model.parameters(), lr=LEARNING_RATE)

# PERFORMANCE: change loss to softmax/cross-entropy loss?
loss_func = nn.MSELoss()

train(train_data[0], enc_model, dec_model, enc_opt, dec_opt, loss_func)
