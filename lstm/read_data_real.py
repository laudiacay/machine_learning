import torchtext
from torchtext import data

SENTENCES = data.Field()

train, val, test = data.TabularDataset.splits(
    path='bobsue-data/',
    train='bobsue.seq2seq.train.tsv',
    validation='bobsue.seq2seq.dev.tsv',
    test='bobsue.seq2seq.test.tsv',
    format='tsv', fields=[('sent_1', SENTENCES), ('sent_2', SENTENCES)]
)
for t in train:
    print(t.sent_1)
