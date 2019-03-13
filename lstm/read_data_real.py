import torchtext
from torchtext import data
import torch

torch.manual_seed(42)

EMBEDDING_SIZE = 200

SENTENCES = data.Field()

train, dev, test = data.TabularDataset.splits(
    path='bobsue-data/',
    train='bobsue.seq2seq.train.tsv',
    validation='bobsue.seq2seq.dev.tsv',
    test='bobsue.seq2seq.test.tsv',
    format='tsv', fields=[('sent_1', SENTENCES), ('sent_2', SENTENCES)]
)

SENTENCES.build_vocab(train, dev, test)

# random embedding
random_embedding = [torch.rand(EMBEDDING_SIZE) / 5.0 - 0.1
                        for _ in range(len(SENTENCES.vocab))]
SENTENCES.vocab.set_vectors(SENTENCES.vocab.stoi, random_embedding, 1)

# pretrained embeddings
SENTENCES.vocab.load_vectors('glove.6B.200d')
