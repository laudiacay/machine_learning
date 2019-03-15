import torch
import read_data
import torch.nn as nn
import torch.optim as optim
import os, time, random, math
from tqdm import tqdm

BEG_TOK = '<s>'
END_TOK = '</s>'

EMB_DIM = 200
HID_DIM = 64
DROPOUT = 0.8
LEARNING_RATE = 1
TEACHER_FORCING = 0.8

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

N_EPOCHS = 50
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model.pth')

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
    def __init__(self, OUT_DIM, SENT):
        super(Seq2Seq, self).__init__()
      
        self.SENT = SENT
       
        self.enc_emb = nn.Embedding(OUT_DIM, EMB_DIM)
        self.enc_emb.weight.data.copy_(SENT.vocab.vectors)
        self.dec_emb = nn.Embedding(OUT_DIM, EMB_DIM)
        self.dec_emb.weight.data.copy_(SENT.vocab.vectors)
        
        self.enc_lstm = nn.LSTM(EMB_DIM, HID_DIM)
        self.enc_dropout = nn.Dropout(DROPOUT)

        self.dec_lstm = nn.LSTM(EMB_DIM, HID_DIM)
        self.dec_dropout = nn.Dropout(DROPOUT)
        
        self.out_layer = nn.Linear(EMB_DIM + HID_DIM * 3, OUT_DIM)
    
    def init_hidden(self, BATCH_SIZE):
        return torch.zeros([BATCH_SIZE, HID_DIM]).unsqueeze(0)

    def forward(self, src, trg, teacher_forcing=TEACHER_FORCING):
        BATCH_SIZE = src.shape[1]
        h_i, c_i = self.init_hidden(BATCH_SIZE), self.init_hidden(BATCH_SIZE)
        
        has_trg = trg is not None
         
        outputs = []
        emb_src = self.enc_dropout(self.enc_emb(src))
        _, (h_i, c_i) = self.enc_lstm(emb_src, (h_i, c_i))
        x_i = trg[0, :] if has_trg else torch.tensor([self.SENT.vocab.stoi[BEG_TOK]])
        h_0, c_0 = h_i, c_i
        i = 1
        while True:
            embedded = self.dec_dropout(self.dec_emb(x_i.unsqueeze(0)))
            output, (hidden, context) = self.dec_lstm(embedded, (h_i, c_i))
            output = self.out_layer(torch.cat((output.squeeze(0), embedded.squeeze(0), h_0.squeeze(0), c_0.squeeze(0)), dim=1))
            outputs.append(output)
            pred = output.max(1)[1]
            teacher_force = random.random() < teacher_forcing
            x_i = trg[i] if teacher_force and has_trg else pred
            i += 1
            if not has_trg: 
                if i > 3 * src.shape[0] or pred == self.SENT.vocab.stoi[END_TOK]: 
                    break
            else: 
                if i >= trg.shape[0]:
                    break
        
        outputs = torch.stack(outputs)
        return outputs

train_i, dev_i, test_i, sent = read_data.load(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)
input_dim = len(sent.vocab)
output_dim = len(sent.vocab)

model = Seq2Seq(output_dim, sent)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
pad_idx = sent.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def calc_accuracy(output, target, pad_ind=0):
    _, preds = output.max(1)
    correct_count = 0
    total_count = 0
    assert(preds.shape[0] >= target.shape[0])
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if target[i,j] != pad_ind and preds[i,j] != pad_ind:
                total_count += 1
                if target[i,j] == preds[i,j]: correct_count += 1
    return correct_count / total_count

def train(model, iterator, optimizer, criterion, clip): 
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in tqdm(iterator):
        src = batch.sent_1
        trg = batch.sent_2
        optimizer.zero_grad()
        output = model(src, trg).permute(0,2,1)
        loss = criterion(output, trg[1:])
        acc = calc_accuracy(output, trg[1:], pad_idx) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def indices_to_words(phrase, sentences):
    return [sentences.vocab.itos[word] for word in phrase]

def eval_once(data_iterable, model, SENTENCES):
    with torch.no_grad():
        for batch in data_iterable:
            output = model.forward(batch.sent_1, None)
            preds = torch.tensor([[o.max(1)[1][0]] for o in output])
            if random.random() < 0.1:
                print('sent_1', indices_to_words(batch.sent_1, SENTENCES))
                print('sent_2', indices_to_words(batch.sent_2, SENTENCES))
                print('preds!', indices_to_words(preds, SENTENCES))

def evaluate(model, iterator, criterion, SENTENCES): 
    model.eval()
    epoch_loss = 0
    epoch_acc = 0 
    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.sent_1
            trg = batch.sent_2
            output = model(src, trg, 0).permute(0,2,1)
            loss = criterion(output, trg[1:])
            acc = calc_accuracy(output, trg[1:], pad_idx) 
            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

train_losses = []
valid_losses = []

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_i, optimizer, criterion, CLIP)
    valid_loss, valid_acc = evaluate(model, dev_i, criterion, sent)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train acc: {train_acc:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. acc: {valid_acc:7.3f}')
print(train_losses)
print(valid_losses)
'''

train_i, dev_i, test_i, sent = read_data.load(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)

INPUT_DIM = len(sent.vocab)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
eval_once(dev_i, model, sent)

'''
