import torchtext
from torchtext import data
import torch

EMBEDDING_SIZE = 200

PAD_TOK = '<pad>'
START_TOK = '<s>'
END_TOK = '</s>'

def build_dataset(device):
    SENTENCES = data.Field(pad_token=PAD_TOK, tensor_type=torch.cuda.LongTensor)

    train, dev, test = data.TabularDataset.splits(
        path='bobsue-data/',
        train='bobsue.seq2seq.train.tsv',
        validation='bobsue.seq2seq.dev.tsv',
        test='bobsue.seq2seq.test.tsv',
        format='tsv', fields=[('sent_1', SENTENCES), ('sent_2', SENTENCES)],
        device=device
    )

    SENTENCES.build_vocab(train, dev, test)

    # random embedding
    random_embedding = [torch.rand(EMBEDDING_SIZE, device=cuda) / 5.0 - 0.1
                            for _ in range(len(SENTENCES.vocab))]
    SENTENCES.vocab.set_vectors(
        SENTENCES.vocab.stoi,
        random_embedding,
        EMBEDDING_SIZE
    )

    # or, pretrained embeddings
    # SENTENCES.vocab.load_vectors('glove.6B.200d')

    return train, dev, test, SENTENCES


def load(device, train_batch_size=32, test_batch_size=256):
    train, dev, test, SENTENCES = build_dataset(device)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test),
        batch_sizes=(train_batch_size, test_batch_size, test_batch_size),
        device=device
    )

    return train_iter, dev_iter, test_iter, SENTENCES
