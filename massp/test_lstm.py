#!/usr/bin/env python3

import numpy as np
import pickle
from argparse import ArgumentParser
from tensorflow.keras import models
from tensorflow.keras.preprocessing import sequence


def parse_cmd_args():
    """
    Parse command-line arguments.

    Returns
    -------
    Parsed command-line arguments.

    """
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--infile', dest='infile', required=True, type=str,
        help='''Input file containing the tokens.'''
    )
    parser.add_argument(
        '-t', '--tokenizer', dest='tokenizer', required=True, type=str,
        help='''Path to the LSTM model.'''
    )
    parser.add_argument(
        '-m', '--model', dest='model', required=True, type=str,
        help='''Path to the LSTM model.'''
    )
    return parser.parse_args()


def main():
    # parse input tokens
    cmd_args = parse_cmd_args()
    with open(cmd_args.infile, 'rt') as ipf:
        token_list = [line.strip() for line in ipf]

    # tokenize test texts
    with open(cmd_args.tokenizer, 'rb') as ipf:
        tokenizer = tokenizer = pickle.load(ipf)
    # tokenizer.fit_on_texts(token_list)
    sequences = tokenizer.texts_to_sequences(token_list)
    X = sequence.pad_sequences(sequences, maxlen=1000)

    # load the LSTM model
    lstm_model = models.load_model(cmd_args.model, compile=False)
    class_prob = lstm_model.predict(X)

    protein_classes = ['Bitopic', 'TM-alpha', 'TM-beta', 'Soluble']
    pred_class = [protein_classes[i] for i in np.argmax(class_prob, axis=1)]

    for p, c in zip(class_prob, pred_class):
        print(p, c)


if __name__ == '__main__':
    main()
