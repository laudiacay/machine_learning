import torchtext
from torchtext import data, vocab

EMBEDDING_SIZE = 200

def read_vocab(vocab_file):
    # puts words from vocab_file into a list of strings
    with open(vocab_file) as f:
        vocab_list = f.read().splitlines()
    return vocab_list

def get_embedding(word):
    return embedding[vocab_ixs[word]]

# list of word strings -> tuple of lists of tensors of embeddings for words
def sentence_list_to_tensors(sentence):
    tensors_1 = []
    tensors_2 = []
    snd_sent_ind = sentence.index('</s>')
    fst_sent = sentence[:snd_sent_ind + 1]
    snd_sent = sentence[snd_sent_ind + 1:]
    for word in fst_sent:
        tensors_1.append(get_embedding(word))
    for word in snd_sent:
        tensors_2.append(get_embedding(word))
    return tensors_1, tensors_2

def all_data_to_tensors(sentences):
    return [sentence_list_to_tensors(sentence) for sentence in sentences]

def get_embedded_data():
    SENTENCES = data.Field()

    train, val, test = data.TabularDataset.splits(
        path='bobsue-data/',
        train='bobsue.seq2seq.train.tsv',
        validation='bobsue.seq2seq.dev.tsv',
        test='bobsue.seq2seq.test.tsv',
        format='tsv', fields=[('sent_1', SENTENCES), ('sent_2', SENTENCES)]
    )
    train_data = all_data_to_tensors(train_raw)
    dev_data = all_data_to_tensors(dev_raw)
    test_data = all_data_to_tensors(test_raw)
    return train_data, dev_data, test_data

def get_word_from_embedded(output):
    dotted = torch.matmul(embedding, output)
    _, ind = dotted.softmax(0).max(0)
    return vocab_lst[ind]

def gen_embedding(LOAD=False):
    global vocab_lst, vocab_ixs, embedding
    vocab_lst = read_vocab('bobsue-data/bobsue.voc.txt')
    vocab_ixs = {k: v for v, k in enumerate(vocab_lst)}
    if LOAD:
        embedding = torch.load('embedding.pth')
    else:
        embedding = (torch.rand([len(vocab_lst), EMBEDDING_SIZE]) / 5.0) - 0.1
        torch.save(embedding, 'embedding.pth')
