# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

KERNEL_SIZE = 3

CHAR_EMBEDDING_DIM = 16
WORD_EMBEDDING_DIM = 16
HIDDEN_DIM = 16
EPOCH_NUM = 2

BATCH_SIZE = 100
NUM_WORKERS = 0

ALPHABET_SIZE = 256

UNKNOWN = ord(' ')  # 32

MAX_WORD_LENGTH = 64


def parse_line(raw_line):
    words = []
    tags = []
    for entry in raw_line.split(' '):
        split = entry.split('/')
        words += split[:-1]
        tags += [split[-1]] * len(split[:-1])
    return words, tags


def parse_file(path):
    sentences = []
    with open(path, 'r') as input_file:
        lines = input_file.read().splitlines()
        for line in lines:
            sentence, tags = parse_line(line)
            sentences.append((sentence, tags))
    return sentences


class POSDataset(Dataset):
    def __init__(self, path):
        sentences, tags = zip(*parse_file(path))
        self.sentence_lengths = [len(sent) for sent in sentences]

        self.word_to_ix, self.tag_to_ix, self.ix_to_tag = POSDataset.build_indices(sentences, tags)
        self.word_encoddings = {}

        self.sentences = [self.encode_sentence(sentence) for sentence in sentences]
        self.tags = [self.encode_tags(sentence_tags) for sentence_tags in tags]

    @staticmethod
    def build_indices(sentences, tags):
        word_to_ix = {}
        tag_to_ix = {}
        ix_to_tag = {}

        word_to_ix[' '] = UNKNOWN

        for sentence, tags in zip(sentences, tags):
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
            for tag in tags:
                if tag not in tag_to_ix:
                    ix_to_tag[len(tag_to_ix)] = tag
                    tag_to_ix[tag] = len(tag_to_ix)
        return word_to_ix, tag_to_ix, ix_to_tag

    def vocabulary_size(self):
        return len(self.word_to_ix.keys())

    def tagset_size(self):
        return len(self.tag_to_ix.keys())

    def pad_chars_encoding(self, word):
        padder = UNKNOWN
        return word + [padder] * (MAX_WORD_LENGTH - len(word))

    def encode_word(self, word):
        """
        :param word: some
        :return: [ CODE(SOME), CODE(S), CODE(O), CODE(M), CODE(E) ]
        """
        if word in self.word_encoddings:
            return self.word_encoddings[word]

        chars_encoding = [ord(c) for c in word]
        chars_encoding = self.pad_chars_encoding(chars_encoding)
        word_encoding = [self.word_to_ix[word]] + chars_encoding
        result = torch.tensor(word_encoding)
        self.word_encoddings[word] = result
        return result

    def max_sent_len(self):
        return max(self.sentence_lengths)

    def pad_sentence_encoding(self, sentence: list) -> list:
        padder = torch.tensor([UNKNOWN] * (MAX_WORD_LENGTH + 1))
        return sentence + [padder] * (self.max_sent_len() - len(sentence))

    def encode_sentence(self, sentence):
        sentence_encoding = [self.encode_word(word) for word in sentence]
        sentence_encoding = self.pad_sentence_encoding(sentence_encoding)
        return torch.stack(sentence_encoding)

    def pad_tags_encoding(self, tags):
        padder = [UNKNOWN]
        return tags + padder * (self.max_sent_len() - len(tags))

    def encode_tags(self, sentence_tags):
        tags_encoding = [self.tag_to_ix[tag] for tag in sentence_tags]
        tags_encoding = self.pad_tags_encoding(tags_encoding)
        return torch.tensor(tags_encoding)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx], self.sentence_lengths[idx]


class CharEmbedding(nn.Module):
    def __init__(self, embedding_dim, kernel_size, alphabet_size=ALPHABET_SIZE):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(alphabet_size, embedding_dim)
        self.char_cnn = nn.Sequential(
            # out channels may be different embedding size
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size),
            nn.ReLU(),
        )

    def forward(self, chars_encoding):
        # (bath size, sentence len, word len)
        bs, sl, wl = chars_encoding.shape
        out = chars_encoding.view(bs * sl, wl)
        out = self.embedding(out)
        out = torch.transpose(out, 1, 2)
        out = self.char_cnn(out)
        out, _ = torch.max(out, dim=2)
        out = out.view(bs, sl, -1)
        return out


class Embedding(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, char_embedding_dim, kernel_size):
        super(Embedding, self).__init__()
        self.char_embeddings = CharEmbedding(char_embedding_dim, kernel_size)
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

    def forward(self, encoding):
        c_emb = self.char_embeddings(encoding[:, :, 1:])
        w_emb = self.word_embeddings(encoding[:, :, 0])
        emb = torch.cat([w_emb, c_emb], dim=2)
        return emb


class LSTMTagger(nn.Module):
    def __init__(self, emb_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, emb):
        lstm_out, _ = self.lstm(emb)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores


class POSTagger(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, kernel_size, hidden_dim,
                 vocab_size, tagset_size):
        super(POSTagger, self).__init__()

        self.embedding = Embedding(vocab_size, word_embedding_dim, char_embedding_dim, kernel_size)
        self.lstm = LSTMTagger(word_embedding_dim + char_embedding_dim, hidden_dim, tagset_size)

    def forward(self, sentence_encoding):
        out = self.embedding(sentence_encoding)
        out = self.lstm(out)
        return out


def loss_func(x, y, l, func=nn.NLLLoss()):
    xs = torch.cat([xi[:li] for xi, li in zip(x, l)])
    ys = torch.cat([yi[:li] for yi, li in zip(y, l)])
    return func(xs, ys)


def train_model(train_file, model_file):
    dataset = POSDataset(train_file)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    pos_tagger = POSTagger(
        WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, KERNEL_SIZE,
        HIDDEN_DIM, dataset.vocabulary_size(), dataset.tagset_size(),
    )
    optimizer = optim.SGD(pos_tagger.parameters(), lr=0.1)

    for epoch in range(EPOCH_NUM):
        pos_tagger.zero_grad()
        for batch in iter(dl):
            sentence_batch, tags_batch, length_batch = batch
            tag_scores = pos_tagger(sentence_batch)
            loss = loss_func(tag_scores, tags_batch, length_batch)
            loss.backward()
            optimizer.step()

        print("Epoch #{} passed. Saving to the model file.".format(epoch))

        conf = dict(WORD_INDEX=dataset.word_to_ix, TAG_INDEX=dataset.tag_to_ix, MODEL_STATE_DICT=pos_tagger.state_dict())
        torch.save(conf, model_file)

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
