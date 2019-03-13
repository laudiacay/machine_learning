import torch

EMBEDDING_SIZE = 200

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
    train_raw = read_tsv('bobsue-data/bobsue.seq2seq.train.tsv')
    dev_raw = read_tsv('bobsue-data/bobsue.seq2seq.dev.tsv')
    test_raw = read_tsv('bobsue-data/bobsue.seq2seq.test.tsv')
    train_data = all_data_to_tensors(train_raw)
    dev_data = all_data_to_tensors(dev_raw)
    test_data = all_data_to_tensors(test_raw)
    return train_data, dev_data, test_data

def get_word_from_embedded(output):
    dotted = torch.matmul(embedding, output)
    _, ind = dotted.softmax(0).max(0)
    return vocab_lst[ind]

# PERFORMANCE: set scale_grad_by_freq = True in embedding?
# PERFORMANCE: should I treat </s> as another dimension?
vocab_lst = read_vocab('bobsue-data/bobsue.voc.txt')
vocab_ixs = {k: v for v, k in enumerate(vocab_lst)}
embedding = (torch.rand([len(vocab_lst), EMBEDDING_SIZE]) / 5.0) - 0.1
