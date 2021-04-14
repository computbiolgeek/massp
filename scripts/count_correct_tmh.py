#!/usr/bin/env python3


"""
    This script counts the number of transmembrane helices in two given
    sequences. One sequence represents the annotations of helices based on
    experimentally determined tertiary structure, the other represents helices
    predicted by MASSP or other peer methods. Only helices that are at least
    10 residues long are considered.
"""

import sys


def main():
    true_file = sys.argv[1]
    pred_file = sys.argv[2]

    with open(true_file, 'rt') as ipf:
        true_str = ''.join([
            line.strip() for line in ipf if not line.startswith('>')
        ])

    # determine TM segment boundaries
    tm_letter = 'M'
    threshold = 10
    tmh_boundaries = []
    prev_res = ''
    start = 0
    end = 0
    for i, x in enumerate(true_str):
        if x == tm_letter and prev_res != tm_letter:
            start = i
        if (x != tm_letter and prev_res == tm_letter) or \
                (x == tm_letter and i == len(true_str) - 1):
            end = i
        if end != 0:
            tmh_boundaries.append((start, end))
            start = 0
            end = 0
        prev_res = x

    with open(pred_file, 'rt') as ipf:
        pred_str = ''.join([
            line.strip() for line in ipf if not line.startswith('>')
        ])

    true_count = 0
    pred_count = 0
    for start, end in tmh_boundaries:
        if end - start >= threshold:
            true_count += 1
        if pred_str[start:end].count(tm_letter) >= threshold:
            pred_count += 1

    print(true_count, pred_count)


if __name__ == '__main__':
    main()
