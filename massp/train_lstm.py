#!/usr/bin/env python3

import os
import  numpy as np
import argparse
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers


def parse_cmd_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    ArgumentParser
        An ArgumentParser object containing the parsed command-line options.

    """
    parser = argparse.ArgumentParser(
        description='''This script can be run to train LSTM models for 
        predicting protein classes.'''
    )
    parser.add_argument(
        '--train-x', dest='train_x', type=str, required=True,
        help='File containing the tokens for training set proteins.'
    )
    parser.add_argument(
        '--train-y', dest='train_y', type=str, required=True,
        help='''File containing the classes labels of training set proteins.'''
    )
    parser.add_argument(
        '--val-x', dest='val_x', type=str, required=True,
        help='''File containing the tokens for validation set proteins.'''
    )
    parser.add_argument(
        '--val-y', dest='val_y', type=str, required=True,
        help='''File containing the class labels of validation set proteins.'''
    )
    parser.add_argument(
        '-m', '--model-path', dest='model_path', type=str, required=False,
        default='./', help='''Path to the directory where the trained model 
        will be stored.'''
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    cmd_args = parse_cmd_arguments()
    train_x = cmd_args.train_x
    train_y = cmd_args.train_y
    val_x = cmd_args.val_x
    val_y = cmd_args.val_y
    model_path = cmd_args.model_path
    
    # train best model
    protein_classes = ['bitopic', 'tm-alpha', 'tm-beta', 'soluble']

    with open(train_x, 'rt') as ipf:
        texts = [l.strip() for l in ipf]
    
    with open(train_y, 'rt') as ipf:
        labels = [l.strip() for l in ipf]
    
    labels = np.array([
        [1 if c == l else 0 for c in protein_classes] for l in labels
    ])
    
    with open(val_x, 'rt') as ipf:
        val_texts = [l.strip() for l in ipf]
    
    with open(val_y, 'rt') as ipf:
        val_labels = [l.strip() for l in ipf]
    
    val_labels = np.array([
        [1 if c == l else 0 for c in protein_classes] for l in val_labels
    ])
    
    max_features = 54
    maxlen = 1000
    batch_size = 4
    
    # tokenize training texts
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen)
    print('Shape of data tensor', data.shape)
    
    # save tokenizer
    with open('tokenizer.pickle', 'wb') as opf:
        pickle.dump(tokenizer, opf, protocol=pickle.HIGHEST_PROTOCOL)

    # tokenize validation texts
    val_tokenizer = Tokenizer(num_words=max_features)
    val_tokenizer.fit_on_texts(val_texts)
    val_sequences = val_tokenizer.texts_to_sequences(val_texts)
    x_val = pad_sequences(val_sequences, maxlen=maxlen)
    y_val = val_labels
    
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    x_train = data[indices]
    y_train = labels[indices]
    
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(16))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    
    model.compile(
        optimizer=optimizers.Adam(
            lr=0.001, 
            beta_1=0.9, 
            beta_2=0.999, 
            amsgrad=False
        ),
        loss='categorical_crossentropy', 
        metrics=['acc']
    )
    
    # train the final model
    model.fit(
        np.concatenate((x_train, x_val)), 
        np.concatenate((y_train, y_val)), 
        epochs=20, 
        batch_size=batch_size
      )
    model.save(os.path.join(model_path, 'final_lstm_model.h5'))


if __name__ == '__main__':
    main()
