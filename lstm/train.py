import torch
cuda = torch.device('cuda')
torch.cuda.manual_seed(42)

from read_data import load, PAD_TOK
from lstm import Seq2Seq

HIDDEN_SIZE = 200
LEARNING_RATE = 1e-3
N_EPOCHS = 15

# TODO: definitely some dimension issues here...
def calc_accuracy(output, target):
    preds = [o.max(1)[1] for o in output]
    correct_list = [o == t for o, t in zip(output[1:], target[1:])]
    return sum(correct_list)/max(len(output) - 1, len(target) - 1)

# TODO: do I need to add clipping to prevent exploding grads?
def train(data_iterable, model, opt, loss_func):
    # turn on dropouts, if you use them
    # model.train()
    epoch_losses = []
    epoch_accuracies = []
    for batch in tqdm(data_iterable):
        opt.zero_grad()
        output = model.forward(batch.sent_1, batch.sent_2)
        loss = loss_func(output, batch.sent_2)
        accuracy = calc_accuracy(output, batch.sent_2)
        loss.backward()
        # TODO: clipping?
        opt.step()
        epoch_losses.append(loss)
        epoch_accuracies.append(accuracy)
    avg_loss = sum(epoch_losses)/len(epoch_losses)
    avg_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
    return avg_loss, avg_accuracy

def eval_model(data_iterable, model, loss_func):
    # turn off dropouts, if you use them
    # model.eval()
    epoch_losses = []
    epoch_accuracies = []
    # speed things up by not calculating gradients, we aren't backpropping
    with torch.no_grad():
        for batch in tqdm(data_iterable):
            output = model.forward(batch.sent_1, None)
            loss = loss_func(output, batch.sent_2)
            accuracy = calc_accuracy(output, batch.sent_2)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
    avg_loss = sum(epoch_losses)/len(epoch_losses)
    avg_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
    return avg_loss, avg_accuracy

def train_main():
    train_iter, dev_iter, test_iter, SENTENCES = load(cuda)
    torch.save(SENTENCES.vocab.vectors, 'vectors.pth')

    model = Seq2Seq(HIDDEN_SIZE, SENTENCES).cuda()

    # PERFORMANCE: Adam for optimization?
    # opt = torch.optim.Adam(dec_model.parameters(), lr=LEARNING_RATE)
    opt = torch.optim.SGD(dec_model.parameters(), lr=LEARNING_RATE)

    # PERFORMANCE: change loss to softmax/cross-entropy loss?
    # loss = nn.CrossEntropyLoss(ignore_index=SENTENCES.vocab.stoi[PAD_TOK])
    loss = nn.MSELoss(ignore_index=SENTENCES.vocab.stoi[PAD_TOK])

    min_dev_loss = float('inf')

    start = time.time()
    for epoch in range(N_EPOCHS):
        training_loss, training_acc = train(train_iter, model, opt, loss)
        dev_loss, dev_acc = eval_model(dev_iter, model, loss)
        end = time.time()

        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            min_dev_acc = dev_acc
            torch.save(model.state_dict(), 'model.pth')

        print(f'Epoch {epoch+1} | Elapsed: {(end - start) / 60:3.3f}m')
        print(f'    Train Loss: {training_loss:.3f}')
        print(f'    Train Acc:  {training_acc:.3f}')
        print(f'    Dev Loss:   {dev_loss:.3f}')
        print(f'    Dev Acc:    {dev_acc:.3f}')
        print()
    print(f'Total Elapsed Time: {(end - start) / 60:3.3f}m')
    print(f'Final Saved Dev Loss: {min_dev_acc:.3f}')
    print(f'Final Saved Dev Acc:  {min_dev_acc:.3f}')

def load_main():
    print('loading data and vector embeddings')
    vectors = torch.load('vectors.pth').to(cuda)
    train_iter, dev_iter, test_iter, SENTENCES = load()
    SENTENCES.vocab.vectors = vectors
    print('loaded data and vector embeddings')
    print('loading model from files')
    model = Seq2Seq(HIDDEN_SIZE, SENTENCES).cuda()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    print('loaded model')

    # TODO: do predictions, print them, get accuracies

if __name__ == '__main__':
    TRAIN = True
    if TRAIN:
        train_main()
    else:
        load_main()
