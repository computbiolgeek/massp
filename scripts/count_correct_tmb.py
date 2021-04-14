#!/usr/bin/env python3

"""
    This script counts the number of transmembrane beta strands in two given
    sequences. One sequence represents the annotations of strands based on
    experimentally determined tertiary structure, the other represents strands
    predicted by MASSP or other peer methods.
"""

import sys


def main():

    true_file = sys.argv[1]
    pred_file = sys.argv[2]
    
    with open(true_file, 'rt') as ipf:
        true_str = ''.join([l.strip() for l in ipf if not l.startswith('>')])
    
    # determine TM segment boundaries
    tmh_boundaries = []
    threshold = 5
    prev_res = ''
    start = 0
    end = 0
    for i, x in enumerate(true_str):
        if x == 'M' and prev_res != 'M':
            start = i
        if (x != 'M' and prev_res == 'M') or (x == 'M' and i == len(true_str) - 1):
            end = i
        if end != 0:
            tmh_boundaries.append((start, end))
            start = 0
            end = 0
        prev_res = x
    
    with open(pred_file, 'rt') as ipf:
        pred_str = ''.join([l.strip() for l in ipf if not l.startswith('>')])
    
    true_count = 0
    pred_count = 0
    for start, end in tmh_boundaries:
        if end - start >= threshold:
            true_count += 1
        if pred_str[start:end].count('M') >= threshold:
            pred_count += 1
    
    print(true_count, pred_count)


if __name__ == '__main__':
    main()
