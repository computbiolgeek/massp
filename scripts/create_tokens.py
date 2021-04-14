#!/usr/bin/env python3

from argparse import ArgumentParser


def make_tokens(infile, infile_format='ciphipsi'):
    """

    Parameters
    ----------
    infile_format
    infile

    Returns
    -------

    """
    # map ciphipsi topology symbols to massp topology symbols
    top_types = {'>': 'U', '<': 'D', '-': 's'}

    # token list
    tokens = []

    # iterate through all residues
    with open(infile, 'rt') as ipf:
        for line in ipf:
            l_stripped = line.strip()
            if l_stripped.startswith('#'):
                continue
            # get targets
            ss, loc, ori, top = l_stripped.split()[3:]
            # skip unresolved residues
            if any([x == 'X' for x in [ss, loc, ori, top]]):
                print('Skipped unresolved residue: {}.'.format(l_stripped))
                continue
            if top not in top_types.keys():
                continue
            if ori == '-':
                ori = 's'
            if loc == 'S':
                loc = 's'
            # make a token
            tokens.append(ss + loc + ori + top_types[top])
    # return all tokens together as a string
    return ' '.join(tokens)


def parse_cmd_args():
    """

    Returns
    -------

    """
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--infile', dest='infile', required=True, type=str,
        help='''Input file.'''
    )
    parser.add_argument(
        '-f', '--format', dest='format', type=str, required=False,
        default='ciphipsi', choices=['ciphipsi', 'fasta'],
        help='''Format of the input file.'''
    )
    parser.add_argument(
        '-o', '--outfile', dest='outfile', required=True, type=str,
        help='''Output file.'''
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    cmd_args = parse_cmd_args()

    #
    infile = cmd_args.infile

    # input file format
    infile_format = cmd_args.format

    #
    tokens = make_tokens(infile, infile_format)

    #
    with open(cmd_args.outfile, 'wt') as opf:
        opf.write(tokens + '\n')


if __name__ == '__main__':
    main()