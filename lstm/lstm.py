import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 200

torch.manual_seed(42)

def read_vocab(vocab_file):
    # puts words from vocab_file into a list of strings
    with open(vocab_file) as f:
        vocab_list = f.read().splitlines()
    return vocab_list

# returns a list of sentences
# where a sentence is represented as a list of word strings
def read_tsv(tsv_file):
    with open(tsv_file) as f:
        examples = f.read().splitlines()
    return [e.split() for e in examples]

def example_to_tensor(lines):
    max_line_len = max([len(l) for l in lines])
    tensor = torch.zeros(max_line_len, len(lines), EMBEDDING_SIZE)


class LSTM(nn.Module): 
    def __init__(self, inp_size, hid_size):
        super(LSTM, self).__init__()
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.Wi = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        self.Wf = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        self.Wo = nn.Linear(self.inp_size + self.hid_size, self.hid_size)
        self.Wg = nn.Linear(self.inp_size + self.hid_size, self.hid_size)

    def forward(self, x_seq, h_0, c_0):
        def one_step(x, h_i, c_i):
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
        
        for x in x_seq.size(0):
            h_t, c_t = one_step(x, h_t, c_t)
            out.append(h_t)
        
        out = torch.cat(out, 0)

        return out, h_t, c_t

# PERFORMANCE: set scale_grad_by_freq = True in embedding?
# PERFORMANCE: change distribution of embedding to uniform on [-0.1, 0.1]?
# PERFORMANCE: set max_norm in embedding?
# PERFORMANCE: should I treat </s> as another embedding?
vocab_lst = read_vocab('bobsue-data/bobsue.voc.txt')
vocab_ixs = {k: v for v, k in enumerate(vocab_lst)}
embedding = nn.Embedding(len(vocab_lst), 200)

train_raw = read_tsv('bobsue-data/bobsue.seq2seq.train.tsv')
dev_raw = read_tsv('bobsue-data/bobsue.seq2seq.dev.tsv')
test_raw = read_tsv('bobsue-data/bobsue.seq2seq.test.tsv')

print(train_raw)

model = LSTM(200, 200)
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# PERFORMANCE: change loss to softmax/cross-entropy loss?
loss_func = nn.MSELoss()

lstm.zero_grad()
loss.backward()
