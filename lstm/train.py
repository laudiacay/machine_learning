import torch
#torch.manual_seed(442)

from read_data import load, PAD_TOK
from lstm import Seq2Seq
import time
from tqdm import tqdm

HIDDEN_SIZE = 64
LEARNING_RATE = 0.5
N_EPOCHS = 25

def calc_accuracy(output, target, pad_ind=0):
    _, preds = output.max(1)
    correct_count = 0
    total_count = 0
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if target[i, j] != pad_ind:
                total_count += 1
                if target[i, j] == preds[i, j]: correct_count += 1
    return correct_count / total_count

def train(data_iterable, model, opt, loss_func, pad_ind=0):
    # turn on dropouts, if you use them
    # model.train()
    epoch_losses = []
    epoch_accuracies = []
    num_batches = len(data_iterable)
    for batch in tqdm(data_iterable):
        opt.zero_grad()
        output = model.forward(batch.sent_1, batch.sent_2)
        loss = loss_func(output, batch.sent_2[1:])
        accuracy = calc_accuracy(output, batch.sent_2[1:], pad_ind=pad_ind)
        loss.backward()
        # TODO: clipping?
        opt.step()
        epoch_losses.append(loss)
        epoch_accuracies.append(accuracy)
    avg_loss = sum(epoch_losses)/len(epoch_losses)
    avg_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
    return avg_loss, avg_accuracy

def pad_out(output, target, pad_ind):
    output = output[:min(output.shape[0], target.shape[0]), :, :]
    new_out = torch.full([target.shape[0], output.shape[1], output.shape[2]], pad_ind)
    new_out[:output.shape[0], :, :] = output
    return new_out

def eval_model(data_iterable, model, loss_func, pad_ind=0):
    # turn off dropouts, if you use them
    # model.eval()
    epoch_losses = []
    epoch_accuracies = []
    # speed things up by not calculating gradients, we aren't backpropping
    with torch.no_grad():
        for batch in tqdm(data_iterable):
            output = model.forward(batch.sent_1, None) 
            output = pad_out(output, batch.sent_2[1:], pad_ind)
            loss = loss_func(output, batch.sent_2[1:])
            accuracy = calc_accuracy(output, batch.sent_2[1:], pad_ind=pad_ind)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
    avg_loss = sum(epoch_losses)/len(epoch_losses)
    avg_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
    return avg_loss, avg_accuracy

def indices_to_words(phrase, SENTENCES):
    return [SENTENCES.vocab.itos[word] for word in phrase]


def eval_once(data_iterable, model, SENTENCES):
    with torch.no_grad():
        for batch in tqdm(data_iterable):
            output = model.forward(batch.sent_1, None)
            _, preds = output.max(1)
            print('sent_1', indices_to_words(batch.sent_1, SENTENCES))
            print('sent_2', indices_to_words(batch.sent_2, SENTENCES))
            print('preds!', indices_to_words(preds, SENTENCES))

def train_main():
    train_iter, dev_iter, test_iter, SENTENCES = load()
    torch.save(SENTENCES.vocab.vectors, 'vectors.pth')

    PAD_IND = SENTENCES.vocab.stoi[PAD_TOK]

    model = Seq2Seq(HIDDEN_SIZE, SENTENCES)

    opt = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    # PERFORMANCE: change loss to softmax/cross-entropy loss?
    loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_IND)
    # loss = torch.nn.MSELoss(ignore_index=SENTENCES.vocab.stoi[PAD_TOK])

    min_dev_loss = float('inf')

    start = time.time()
    for epoch in range(N_EPOCHS):
        training_loss, training_acc = train(train_iter, model, opt, loss,
                pad_ind=PAD_IND)
        dev_loss, dev_acc = eval_model(dev_iter, model, loss, pad_ind=PAD_IND)
        end = time.time()

        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            min_dev_acc = dev_acc
            torch.save(model.state_dict(), 'model.pth')

        print('Epoch {} | Elapsed: {:3.3}m'.format(epoch+1, (end - start) / 60))
        print('    Train Loss: {:.3}'.format(training_loss))
        print('    Train Acc:  {:.3}'.format(training_acc))
        print('    Dev Loss:   {:.3}'.format(dev_loss))
        print('    Dev Acc:    {:.3}'.format(dev_acc))
        print()
    print('Total Elapsed Time: {:3.3}m'.format((end - start) / 60))
    print('Final Saved Dev Loss: {:.3}'.format(min_dev_loss))
    print('Final Saved Dev Acc:  {:.3}'.format(min_dev_acc))

def load_main():
    print('loading data and vector embeddings')
    vectors = torch.load('vectors.pth')
    train_iter, dev_iter, test_iter, SENTENCES = load(train_batch_size=1)
    SENTENCES.vocab.vectors = vectors
    print('loaded data and vector embeddings')
    print('loading model from files')
    model = Seq2Seq(HIDDEN_SIZE, SENTENCES)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print('loaded model')
    #eval_once(dev_iter, model, SENTENCES) 
    eval_once(train_iter, model, SENTENCES)
    # TODO: do predictions, print them, get accuracies

if __name__ == '__main__':
    TRAIN = True
    if TRAIN:
        train_main()
    else:
        load_main()
