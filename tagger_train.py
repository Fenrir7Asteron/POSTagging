# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ALPHABET_SIZE = 256
CHAR_EMBEDDING_DIM = 16
WORD_EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EPOCH_NUM = 20


# def prepare_sequence(seq, word_to_ix, char_to_ix):
#     idxs = []
#     for word in seq:
#         word_idx = word_to_ix[word]
#         char_idxs = [char_to_ix[char] for char in word]
#         idxs.append((word_idx, char_idx))
#
#     return torch.tensor(idxs, dtype=torch.long)


def prepare_sequence(seq, to_ix, item_len):
    # todo: padding index should be never used for other item
    idxs = [to_ix[w] for w in seq] + [0] * (item_len - len(seq))
    return torch.tensor(idxs, dtype=torch.long)


def load_data(path):
    training_data = []
    with open(path, "r") as input_file:
        lines = input_file.read().splitlines()
        for line in lines:
            sentences = []
            tags = []
            for token in line.split(' '):
                split = token.split('/')
                sentences += split[:-1]
                tags += [split[-1]] * len(split[:-1])
            training_data.append((sentences, tags))

    word_to_ix = {}
    char_to_ix = {}
    tag_to_ix = {}
    ix_to_tag = {}

    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                ix_to_tag[len(tag_to_ix)] = tag
                tag_to_ix[tag] = len(tag_to_ix)

    print(word_to_ix)
    print(char_to_ix)
    print(tag_to_ix)
    print(ix_to_tag)

    return training_data, word_to_ix, char_to_ix, tag_to_ix, ix_to_tag


class CharEmbedding(nn.Module):
    def __init__(self, alphabet_size, embedding_dim, kernel_size):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(alphabet_size, embedding_dim)
        self.char_cnn = nn.Sequential(
            # out channels may be different embedding size
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size),
            nn.ReLU(),
        )

    def forward(self, word):
        out = self.embedding(word)
        print(out.shape)
        out = torch.transpose(out, 1, 2)
        print(out.shape)
        out = self.char_cnn(out)
        print(out.shape)
        out, _ = torch.max(out, dim=2)
        print(out.shape)
        return out


class LSTMTagger(nn.Module):
    def __init__(self, char_embedding_dim, word_embedding_dim, hidden_dim, alphabet_size, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.onehot_encodding = nn.functional.one_hot
        self.char_embedding = CharEmbedding(alphabet_size, char_embedding_dim, 3)
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        # print(sentence)
        # onehot_encoddings = self.onehot_encodding(sentence)
        # print(onehot_encoddings.shape)
        # char_embeddings = self.char_cnn(onehot_encoddings)
        # print(char_embeddings.shape)
        word_embeddings = self.word_embeddings(sentence)
        word_embeddings = word_embeddings.view(len(sentence), 1, -1)

        lstm_out, _ = self.lstm(word_embeddings)
        lstm_out = lstm_out.view(len(sentence), -1)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def train_model(train_file, model_file):
    training_data, word_to_ix, char_to_ix, tag_to_ix, ix_to_tag = load_data(train_file)
    max_word_len = max([len(w) for w in word_to_ix.keys()])

    char_emb = CharEmbedding(len(char_to_ix), CHAR_EMBEDDING_DIM, 3)

    model = LSTMTagger(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, HIDDEN_DIM, ALPHABET_SIZE, len(word_to_ix),
                       len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(EPOCH_NUM):
        for sentence, tags in training_data:
            model.zero_grad()

            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # print(sentence_in)
            # targets = prepare_sequence(tags, tag_to_ix)
            s = torch.stack([prepare_sequence(word, char_to_ix, max_word_len) for word in sentence])
            print(s.shape)
            t = char_emb(s)
            print('PPPPP', t.shape)
            # tag_scores = model(sentence_in)

            # loss = loss_function(tag_scores, targets)
            # loss.backward()
            # optimizer.step()
        print("Epoch #{} passed. Saving to the model file.".format(epoch))

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)

    torch.save(model.state_dict(), model_file)

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
