import torch
import random

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

# list of word strings -> list of tensors of embeddings for words
def sentence_list_to_tensors(sentence):
    tensors = []
    for word in sentence:
        tensors.append(embedding[vocab_ixs[word]])
    return tensors

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
    _, ind = torch.nn.functional.softmax(dotted, dim=0).max()
    return vocab_lst[ind]
# PERFORMANCE: set scale_grad_by_freq = True in embedding?
# PERFORMANCE: change distribution of embedding to uniform on [-0.1, 0.1]?
# PERFORMANCE: set max_norm in embedding?
# PERFORMANCE: should I treat </s> as another dimension?
vocab_lst = read_vocab('bobsue-data/bobsue.voc.txt')
vocab_ixs = {k: v for v, k in enumerate(vocab_lst)}
embedding = (torch.rand([len(vocab_lst), EMBEDDING_SIZE]) / 5.0) - 0.1

print(vocab_lst[1400])
print(get_word_from_embedded(embedding[1400]))
