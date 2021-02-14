# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import multiprocessing

KERNEL_SIZE = 3

CHAR_EMBEDDING_DIM = 16
WORD_EMBEDDING_DIM = 32
HIDDEN_DIM = 32

BATCH_SIZE = 10
NUM_WORKERS = 0

ALPHABET_SIZE = 256

UNKNOWN = ord(' ')  # 32

MAX_WORD_LENGTH = 64


def parse_file(path):
    sentences = []
    with open(path, 'r') as input_file:
        lines = input_file.read().splitlines()
        for line in lines:
            sentence = line.split()
            sentences.append(sentence)
    return sentences


class POSTestDataset(Dataset):
    def __init__(self, path, word_to_ix, tag_to_ix):
        sentences = parse_file(path)
        self.sentence_lengths = [len(sent) for sent in sentences]

        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.word_encoddings = {}

        self.sentences = [self.encode_sentence(sentence) for sentence in sentences]

    def vocabulary_size(self):
        return len(self.word_to_ix.keys())

    def tagset_size(self):
        return len(self.tag_to_ix.keys())

    def pad_chars_encoding(self, word):
        padder = UNKNOWN
        # In case word is longer than any of the training dataset
        # it is truncated to the known maximum word length
        padded = word + [padder] * (MAX_WORD_LENGTH - len(word))
        return padded[:MAX_WORD_LENGTH]

    def encode_word(self, word):
        """
        :param word: some
        :return: [ CODE(SOME), CODE(S), CODE(O), CODE(M), CODE(E) ]
        """
        if word in self.word_encoddings:
            return self.word_encoddings[word]

        chars_encoding = [ord(c) for c in word]
        chars_encoding = self.pad_chars_encoding(chars_encoding)
        w_code = self.word_to_ix[word] if word in self.word_to_ix else UNKNOWN
        word_encoding = [w_code] + chars_encoding
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

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.sentence_lengths[idx]


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


#
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


def tag_sentence(test_file, model_file, out_file):
    conf = torch.load(model_file)
    word_to_ix = conf['WORD_INDEX']
    tag_to_ix = conf['TAG_INDEX']
    model_state_dict = conf['MODEL_STATE_DICT']


    dataset = POSTestDataset(test_file, word_to_ix, tag_to_ix)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    pos_tagger = POSTagger(
        WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, KERNEL_SIZE,
        HIDDEN_DIM, dataset.vocabulary_size(), dataset.tagset_size()
    )

    ix_to_tag = {v:k for k, v in tag_to_ix.items()}
    ix_to_word = {v:k for k, v in word_to_ix.items()}

    pos_tagger.load_state_dict(model_state_dict)
    pos_tagger.zero_grad()
    pos_tagger.eval()
    sentences =[]
    predictions = []

    with torch.no_grad():
        for batch in iter(dl):
            sentence_batch, len_batch = batch
            tag_scores = pos_tagger(sentence_batch)
            outputs = torch.argmax(tag_scores, dim=2)

            for sent in outputs:
                predictions.append([ix_to_tag[w.tolist()] for w in sent])

            sentences+=sentence_batch

    with open(out_file, "w") as output_file:
        output_file.truncate(0)
        for sent, tags in zip(sentences, predictions):
            line = ' '.join([f'{ix_to_word[word[0].tolist()]}/{tag}' for word, tag in zip(sent, tags)]) + '\n'
            output_file.write(line)

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
