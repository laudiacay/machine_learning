import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

import read_data as rd
import random, time
from tqdm import tqdm

LEARNING_RATE = 0.005


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
        if x_seq is not None:
            for x in x_seq:
                h_t, c_t = one_step(x, h_t, c_t)
                out.append(h_t)
        else:
            x = rd.get_embedding('<s>')
            while True:
                h_t, c_t = one_step(x, h_t, c_t)
                out.append(h_t)
                # IS THIS CORRECT?!
                x = h_t 
                pred = rd.get_word_from_embedded(h_t)
                print(pred)
                if pred == '</s>':
                    break
        return out, (h_t, c_t)

    def init_hidden(self):
        return torch.zeros(self.hid_size)

def train(sentences, enc_model, dec_model, enc_opt, dec_opt, criterion):
    sent_1, sent_2 = sentences
    _, (hidden, ctx) = enc_model.forward(sent_1, None, None)
    output, _ = dec_model.forward(sent_2, hidden.detach(), ctx.detach())

    #preds = [rd.get_word_from_embedded(out) for out in output]
    #print(preds)

    enc_opt.zero_grad()
    dec_opt.zero_grad()
    loss = 0
    total_accuracy = 0
    for out, exp_out in zip(output, sent_2):
        loss += criterion(out, exp_out)
        total_accuracy += int(rd.get_word_from_embedded(out)\
                            == rd.get_word_from_embedded(exp_out))
    accuracy = total_accuracy / min(len(output), len(sent_2))
    loss.backward()
    enc_opt.step()
    dec_opt.step()
    return output, loss, accuracy

def predict(sent_1, enc_model, dec_model):
    _, (hidden, ctx) = enc_model(sent_1, None, None)
    output, _ = dec_model.forward(None, hidden.detach(), ctx.detach())
    return output

def predict_data(sentence, enc_model, dec_model, debug=False):
    output = predict(sentence[0], enc_model, dec_model)
    total_accuracy = 0

    if debug:
        in_sent = [rd.get_word_from_embedded(w) for w in sentence[0]]
        print('input:', ' '.join(in_sent))
        exp_out = [rd.get_word_from_embedded(w) for w in sentence[1]]
        print('exp_out:', ' '.join(exp_out))
        preds = [rd.get_word_from_embedded(out) for out in output]
        print('predictions:', ' '.join(preds))

    for out, exp_out in zip(output, sentence[1]):
        total_accuracy += int(rd.get_word_from_embedded(out)\
                            == rd.get_word_from_embedded(exp_out))
    accuracy = total_accuracy / min(len(output), len(sent_2))
    return preds, output, accuracy

def train_main():
    train_data, dev_data, test_data = rd.get_embedded_data()

    enc_model = LSTM(200, 200)
    dec_model = LSTM(200, 200)

    # PERFORMANCE: torch.optim.Adam?
    enc_opt = torch.optim.SGD(enc_model.parameters(), lr=LEARNING_RATE)
    dec_opt = torch.optim.SGD(dec_model.parameters(), lr=LEARNING_RATE)

    # PERFORMANCE: change loss to softmax/cross-entropy loss?
    loss_func = nn.MSELoss()

    N_ITERS = 100000
    logint = 10000
    losses = []
    all_losses = []
    accuracies = []
    start = time.time()

    for i in tqdm(range(N_ITERS + 1)):
        training_example = random.choice(train_data)
        out, loss, accuracy = train(training_example, enc_model, dec_model, enc_opt, dec_opt, loss_func)
        losses.append(loss)
        all_losses.append(loss)
        accuracies.append(accuracy)
        if i % logint == 0:
            elapsed = (time.time() - start) / 60
            avg_loss = sum(losses)/len(losses)
            acc = sum(accuracies)/len(accuracies)
            print('Iter {:7} | Loss: {:5.3f} | Acc: {:.3f} | Elapsed: {:.2f}min'.format(i, avg_loss, acc, elapsed))
            accuracies = []
            losses = []

    torch.save(enc_model.state_dict(), 'enc_model.pth')
    torch.save(dec_model.state_dict(), 'dec_model.pth')

def load_main():
    print('loading models from files!')
    enc_model = LSTM(200, 200)
    enc_model.load_state_dict(torch.load('enc_model.pth'))
    enc_model.eval()
    dec_model = LSTM(200, 200)
    dec_model.load_state_dict(torch.load('dec_model.pth'))
    dec_model.eval()
    print('loaded models.')
    train_data, dev_data, test_data = rd.get_embedded_data()
    # (test_data[:10])
    all_accuracies = []
    print(len(dev_data))
    for i in tqdm(range(len(dev_data))):
        dev = dev_data[i]
        if i % 100 == 0: debug = True
        else: debug = False
        out, acc = predict_data(dev, enc_model, dec_model, debug=debug)
        all_accuracies.append(acc)
    print('overall dev accuracy:', sum(acc)/len(acc))

if __name__ == '__main__':
    TRAIN = True
    rd.gen_embedding(LOAD=not TRAIN)
    if TRAIN:
        train_main()
    else:
        load_main()
