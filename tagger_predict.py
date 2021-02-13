# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import sys
import torch
import numpy as np
import pickle


def load_test_data(path):
    test_data = []
    with open(path, "r") as input_file:
        lines = input_file.read().splitlines()
        for line in lines:
            test_data.append(line.split(' '))
    return test_data


def preprocess_test_data(test_data):
    for i in range(len(test_data)):
        sent = test_data[i]
        for j in range(len(sent)):
            word = test_data[i][j]
            if word not in word_to_ix:
                new_word = random.choice(list(word_to_ix.keys()))
                test_data[i][j] = new_word


def tag_sentence(test_file, model_file, out_file):
    test_data_raw = load_test_data(test_file)
    test_data = preprocess_test_data(test_data_raw)
    model = model.load_state_dict(torch.load(model_file))

    model.zero_grad()
    model.eval()
    test_result = []
    with torch.no_grad():
        for sentence in test_data:
            inputs = prepare_sequence(sentence, word_to_ix)
            tag_scores = model(inputs)
            outputs = torch.argmax(tag_scores, dim=1).tolist()
            tags = [ix_to_tag[w] for w in outputs]
            test_result.append(tags)

    with open(out_file, "w") as output_file:
        output_file.truncate(0)
        for i in range(len(test_data)):
            sentence = test_data[i]
            output_line = ""
            for j in range(len(sentence)):
                word = sentence[j]
                tag = test_result[i][j]
                output_line += '/'.join([word, tag]) + ' '
                print('/'.join([word, tag]), end=" ")
            output_file.write(output_line + "\n")

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
