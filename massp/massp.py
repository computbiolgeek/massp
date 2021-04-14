#!/usr/bin/env python3

"""
MASSP is a two-tier multitasking deep neural network for secondary structure
and transmembrane topology prediction. It makes no a priori assumption about
what protein class the input sequence belongs to. Thus, it is applicable to 
the whole proteome of any organism. Our benchmarking analysis indicates that
MASSP has superior performance to some of the best existing methods: PSIPRED,
NetSurfP-2.0, RaptorX-Property, SPINE-X, OCTOPUS, MEMSAT3, TMHMM2, BOCTOPUS,
PRED-TMBB2, TOPCONS2, and TMP-SS. We're currently preparing a manuscript on 
this method.

@author: Bian Li
@email: bian.li@vanderbilt.edu; comput.biol.geek@gmail.com
"""

import os
import sys
import pickle
from argparse import ArgumentParser

import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# global variables
SEQ_DB = os.environ['SEQ_DB']
HHBLITS = os.environ['HHBLITS']
REFORMAT = os.environ['REFORMAT']
BCL = os.environ['BCL']


def run_hhblits(fasta_file, prefix, ncpu='1'):
    """
    Searches sequences that are homologous to the query sequence using
    the HHblits software suite. 

    Parameters
    ----------
    fasta_file : str
        Protein sequence file in fasta format.
    prefix : str
        Prefix that is to be prepended to all output filenames.
    ncpu : str, optional
        Number of CPUs to run HHblits. The default is '1'.

    Returns
    -------
    None.

    """
    hhblits_options = [
        '-i', fasta_file,
        '-d', SEQ_DB,
        '-e', '0.001',
        '-n', '2',
        '-diff', 'inf',
        'realign_max', '500000',
        '-pre_evalue_thresh', '0.01',
        '-oa3m', prefix + '.a3m',
        '-o', prefix + '.hhr',
        '-ohhm', prefix + '.hhm',
        '-cpu', ncpu
    ]
    
    hhblits_cmd = ' '.join([HHBLITS] + hhblits_options)
    os.system(hhblits_cmd)
    
    # reformatting multiple sequence alignment
    reformat_cmd = ' '.join(
        [
            REFORMAT, '-r',
            'a3m', 'clu',
            prefix + '.a3m',
            prefix + '.clu'
        ]
    )
    os.system(reformat_cmd)
    
    # generate a position-specific scoring matrix
    pssm_cmd = ' '.join(
        [
            BCL, 'bioinfo:MSA2PSSM',
            prefix + '.clu',
            prefix + '.pssm',
            '-efractional_pseudocount', '1',
            '-pseudocount', '1',
            '-msa_background',
            '-fractional_pseudocount', '0.15'
        ]
    )
    os.system(pssm_cmd)


def read_pssm(pssm_file):
    """
    Read a given position-specific scoring matrix into 2D NumPy Array.

    Parameters
    ----------
    pssm_file : str
        File containing a position-specific scoring matrix for the
        input sequence

    Returns
    -------
    pssm_rows : NumPy Array
        PSSM stored in a NumPy Array data structure 
        
    """
    pssm_rows = []
    with open(pssm_file, 'rt') as ipf:
        for l in ipf:
            # skip empty lines or comment lines
            l_stripped = l.strip()
            if not l_stripped or l_stripped.startswith('#') or \
            l_stripped[0].isalpha():
                continue
            pssm_rows.append([float(x) for x in l_stripped.split()[2:-2]])
    return pssm_rows


def create_pssm_tensor(pssm, half_window_size=7):
    """
    Create a NumPy ndarray from the given list of PSSM rows.
    

    Parameters
    ----------
    pssm : list
        A list expected from the read_pssm() method.
    half_window_size : int, optional
        Half-window size or the number of residues to be
        considered on each side of the residue of interest.
        The default is 7.

    Returns
    -------
    pssm_tensor : NumPy ndarray
        A NumPy ndarray of the shape [n_residues, 21, 20, 1]
        that will be used as input to the CNN model.

    """
    n_res = len(pssm)
    pssm_tensor = []
    for i in range(n_res):
        start_idx = i - half_window_size
        end_idx = i + half_window_size + 1
        if start_idx < 0:
            pssm_tensor.append(
                pssm[start_idx:] + pssm[:end_idx]
            )
        elif end_idx > n_res:
            pssm_tensor.append(
                pssm[start_idx:] + pssm[:end_idx-n_res]
            )
        else:
            pssm_tensor.append(
                pssm[start_idx:end_idx]
            )
    return np.array(pssm_tensor)[:, :, :, np.newaxis]


def convert_to_fasta(string, width=60):
    """
    Converts the given string to FASTA style.
    

    Parameters
    ----------
    string : str
        A string that is the output from the 2D-CNN model.
    width : int, optional
        Customizable line width for the FASTA format. 
        The default is 60.

    Returns
    -------
    str
        A string formatted to FASTA style.

    """
    n_lines = len(string) // width
    fasta_formatted = []
    for i in range(n_lines):
        fasta_formatted.append(string[i * width:(i + 1) * width])
    fasta_formatted.append(string[n_lines * width:])
    return '\n'.join(fasta_formatted)


def post_process_sse(sse_pred):
    """
    Remove potential noise in secondary structure elements predicted
    by a trained 2D-CNN.

    Parameters
    ----------
    sse_pred : str
        Secondary structure elements predicted by the 2D-CNN.

    Returns
    -------
    str
        Denoised predicted secondary structure elements.

    """
    for p in ['CHC', 'CCHHCC', 'CEC', 'CCEECC']:
        sse_pred = sse_pred.replace(p, 'C' * len(p))
    for p in ['HCH']:
        sse_pred = sse_pred.replace(p, 'H' * len(p))
    for p in ['ECE', 'EECCEE']:
        sse_pred = sse_pred.replace(p, 'E' * len(p))
    return sse_pred


def parse_loc_pred(loc_pred):
    """
    
    Parameters
    ----------
    loc_pred

    Returns
    -------

    """
    # de-noise
    for p in ['ssMss', 'ssMMss', 'ssMMMMss']:
        loc_pred = loc_pred.replace(p, 's' * len(p))
    for p in ['MMsMM', 'MMssMM', 'MMsssMM']:
        loc_pred = loc_pred.replace(p, 'M' * len(p))

    start_idx = 0
    end_idx = 0
    prev_letter = ''
    segment_info = []
    for i, x in enumerate(loc_pred):
        if x == 'M' and prev_letter != 'M':
            start_idx = i
        if (x != 'M' and prev_letter == 'M') or \
                (x == 'M' and i == len(loc_pred) - 1):
            end_idx = i
        if end_idx != 0:
            tm_bounds = [(start_idx, end_idx)]
            tm_segment = loc_pred[start_idx:end_idx]
            if len(tm_segment) >= 40:  # TM segment too long
                tm_bounds = [
                    (start_idx, start_idx + len(tm_segment) // 2),
                    (start_idx + len(tm_segment) // 2 + 1, end_idx)
                ]
            for s, e in tm_bounds:
                segment_info.append((s, e))
            start_idx = 0
            end_idx = 0
        prev_letter = x
    return segment_info


def post_process_loc(loc_pred):
    """

    Parameters
    ----------
    loc_pred

    Returns
    -------

    """
    segment_info = parse_loc_pred(loc_pred)
    new_pred = 's' * len(loc_pred)
    for i, x in enumerate(segment_info):
        s, e = x
        if e - s < 5:
            continue
        new_pred = new_pred[:s] + 'M' * (e - s) + new_pred[e:]
    return new_pred


def replace_top_segment(top_str, start, end):
    """

    Parameters
    ----------
    top_str
    start
    end

    Returns
    -------

    """
    tm_letters = ['U', 'D']
    top_segment = top_str[start:end]
    u_count = top_segment.count(tm_letters[0])
    d_count = top_segment.count(tm_letters[1])
    if u_count > d_count:
        new_letter = tm_letters[0]
    else:
        new_letter = tm_letters[1]
    return top_str[:start] + new_letter * len(top_segment) + \
               top_str[end:]


def parse_top_pred(top_pred):
    """

    Parameters
    ----------
    top_pred

    Returns
    -------

    """
    # de-noise
    for p in ['sUs', 'sUUs', 'sDs', 'sDDs']:
        top_pred = top_pred.replace(p, 's' * len(p))
    tm_letters = ['U', 'D']
    start_idx = 0
    end_idx = 0
    prev_letter = ''
    segment_info = []
    for i, x in enumerate(top_pred):
        if x in tm_letters and prev_letter not in tm_letters:
            start_idx = i
        if (x not in tm_letters and prev_letter in tm_letters) or \
                (x in tm_letters and i == len(top_pred) - 1):
            end_idx = i
        if end_idx != 0:
            tm_bounds = [(start_idx, end_idx)]
            tm_segment = top_pred[start_idx:end_idx]
            if len(tm_segment) > 35:  # TM segment too long
                tm_bounds = [
                    (start_idx, start_idx + len(tm_segment) // 2),
                    (start_idx + len(tm_segment) // 2 + 1, end_idx)
                ]
            for s, e in tm_bounds:
                tm_letters = ['U', 'D']
                top_segment = top_pred[s:e]
                u_count = top_segment.count(tm_letters[0])
                d_count = top_segment.count(tm_letters[1])
                segment_info.append((s, e, u_count, d_count))
            start_idx = 0
            end_idx = 0
        prev_letter = x
    return segment_info


def post_process_top(top_pred):
    """

    Parameters
    ----------
    top_pred

    Returns
    -------

    """
    segment_info = parse_top_pred(top_pred)
    new_pred = 's' * len(top_pred)
    for i, x in enumerate(segment_info):
        s, e, u, d = x
        if e - s < 5:
            continue
        if i < len(segment_info) - 1:
            _, _, next_u, next_d = segment_info[i + 1]
        if u > d:
            if i == len(segment_info) - 1:
                new_letter = 'U'
            elif next_u < next_d:
                new_letter = 'U'
            else:
                new_letter = 'D'
        else:
            if i == len(segment_info) - 1:
                new_letter = 'D'
            elif next_d < next_u:
                new_letter = 'D'
            else:
                new_letter = 'U'
        new_pred = new_pred[:s] + new_letter * (e - s) + new_pred[e:]
    return new_pred


def parse_cmd_args():
    """
    Parse command line arguments.

    Returns
    -------
    args : ArgumentParser
        ArgumentParser object containing parsed command-line arguments.

    """
    parser = ArgumentParser(description="")
    
    parser.add_argument(
        '--query', '-q', dest='query', type=str, required=True,
        help='''An amino acid sequence in FASTA format for which secondary
        structures and transmembrane topologies are to be predicted.'''
    )
    parser.add_argument(
        '--pssm', '-p', dest='pssm', type=str, required=False,
        help='''Position specific scoring matrix obtained from 
        running the BCL pssm application.'''
    )
    parser.add_argument(
        '--cnn-model', dest='cnn_model', type=str, required=True,
        help='''A multitasking deep CNN model taking PSSM as input 
        and trained to predict secondary structures and transmembrane
        topologies.'''
    )
    parser.add_argument(
        '--lstm-model', dest='lstm_model', type=str, required=True,
        help='''A LSTM model taking predictions from the CNN model as input 
        and trained to predict what protein class the sequence belongs to.'''
    )
    parser.add_argument(
        '--tokenizer', dest='tokenizer', type=str, required=True,
        help='''Tokenizer saved from fitting on training set tokens.'''
    )
    parser.add_argument(
        '--output-prefix', '-o', dest='prefix', type=str, required=False,
        default='out', help='''Prefix that will be prepended to the filenames
        of all output files.'''
    )
    
    args = parser.parse_args()
    return args


def main():
    # get command-line arguments
    cmd_args = parse_cmd_args()
    
    if cmd_args.pssm is None:
        # create a PSSM for the query sequence
        run_hhblits(cmd_args.query, cmd_args.prefix)
    
        # read the PSSM into a NumPy array
        if not os.path.exists(cmd_args.prefix + '.pssm'):
            print(
                '''No position-specific-scoring matrix found!
                Please check the log file.'''
            )
            sys.exit(1)
        pssm = read_pssm(cmd_args.prefix + '.pssm')
    else:
        pssm = read_pssm(cmd_args.pssm)
    
    # create tensors by taking windows from the PSSM array
    pssm_tensor = create_pssm_tensor(pssm, half_window_size=10)
    
    # load the cnn model
    cnn_model = models.load_model(cmd_args.cnn_model, compile=False)
    
    # feature labels
    sse = ['H', 'E', 'C']
    loc = ['M', 's']
    ori = ['L', 'P', 's']
    top = ['U', 'D', 's']
    
    # make predictions
    sse_prob, loc_prob, ori_prob, top_prob = cnn_model.predict(pssm_tensor)
    
    pred_sses = ''.join([sse[i] for i in np.argmax(sse_prob, axis=1)])
    pred_locs = ''.join([loc[i] for i in np.argmax(loc_prob, axis=1)])
    pred_oris = ''.join([ori[i] for i in np.argmax(ori_prob, axis=1)])
    pred_tops = ''.join([top[i] for i in np.argmax(top_prob, axis=1)])

    # post-process predicted residue locations and topologies
    pred_sses = post_process_sse(pred_sses)
    pred_locs = post_process_loc(pred_locs)
    pred_tops = post_process_top(pred_tops)

    combined_predictions = []
    for a, b, c, d in zip(pred_sses, pred_locs, pred_oris, pred_tops):
        combined_predictions.append(a + b + c + d)
    sentence = [' '.join(combined_predictions)]

    # tokenize test texts
    with open(cmd_args.tokenizer, 'rb') as ipf:
        tokenizer = pickle.load(ipf)
    sequences = tokenizer.texts_to_sequences(sentence)
    X = pad_sequences(sequences, maxlen=1000)

    # load the LSTM model
    lstm_model = models.load_model(cmd_args.lstm_model, compile=False)
    class_prob = lstm_model.predict(X)

    protein_classes = ['Bitopic', 'TM-alpha', 'TM-beta', 'Soluble']
    pred_class = [protein_classes[i] for i in np.argmax(class_prob, axis=1)]

    # write predictions to disk files
    with open(cmd_args.prefix + '_predicted_sse.fasta', 'wt') as opf:
        opf.write('>' + 'MASSP-predicted secondary structures\n')
        opf.write(convert_to_fasta(pred_sses))
        opf.write('\n')
    with open(cmd_args.prefix + '_predicted_loc.fasta', 'wt') as opf:
        opf.write('>' + 'MASSP-predicted residue locations\n')
        opf.write(convert_to_fasta(pred_locs))
        opf.write('\n')
    with open(cmd_args.prefix + '_predicted_ori.fasta', 'wt') as opf:
        opf.write('>' + 'MASSP-predicted residue orientations\n')
        opf.write(convert_to_fasta(pred_oris))
        opf.write('\n')
    with open(cmd_args.prefix + '_predicted_top.fasta', 'wt') as opf:
        opf.write('>' + 'MASSP-predicted residue topologies\n')
        opf.write(convert_to_fasta(pred_tops))
        opf.write('\n')
    with open(cmd_args.prefix + '_predicted_tokens.fasta', 'wt') as opf:
        opf.write('>' + 'MASSP-predicted tokens\n')
        tokens = ' '.join([
            a + b + c + d for a, b, c, d in 
            zip(pred_sses, pred_locs, pred_oris, pred_tops)
        ])
        opf.write(tokens)
        opf.write('\n')
        
    # print a message
    print('The given sequence was predicted to be ' + ' '.join(pred_class) + '.')
    print('Residue-level predictions have been written to these files:')
    print(cmd_args.prefix + '_predicted_sse.fasta')
    print(cmd_args.prefix + '_predicted_loc.fasta')
    print(cmd_args.prefix + '_predicted_ori.fasta')
    print(cmd_args.prefix + '_predicted_top.fasta')
    print(cmd_args.prefix + '_predicted_tokens.fasta')
    print('Thanks for using MASSP!')


if __name__ == '__main__':
    main()
