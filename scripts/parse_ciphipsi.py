#!/usr/bin/env python3

from argparse import ArgumentParser


def parse_ciphipsi(infile):
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
    sses = []
    locs = []
    oris = []
    tops = []
    
    # iterate through all residues
    with open(infile, 'rt') as ipf:
        for line in ipf:
            l_stripped = line.strip()
            if l_stripped.startswith('#'):
                continue
            # get targets
            ss, loc, ori, top = l_stripped.split()[3:]
            # skip unresolved residues
            if ss in ['H', 'E', 'C']:
                sses.append(ss)
            if loc in ['M', 'S']:
                if loc == 'S':
                    loc = 's'
                locs.append(loc)
            if ori in ['L', 'P', '-']:
                if ori == '-':
                    ori = 's'
                oris.append(ori)
            if top in top_types.keys():
                tops.append(top_types[top])
    # return all tokens together as a string
    return ''.join(sses), ''.join(locs), ''.join(oris), ''.join(tops)


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
        '-p', '--prefix', dest='prefix', required=True, type=str,
        help='''Output file.'''
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    cmd_args = parse_cmd_args()

    #
    infile = cmd_args.infile

    #
    sses, locs, oris, tops = parse_ciphipsi(infile)

    #
    with open(cmd_args.prefix + '_true_sses.fasta', 'wt') as opf:
        opf.write('>' + cmd_args.prefix + ' true sses\n')
        opf.write(sses + '\n')

    with open(cmd_args.prefix + '_true_locs.fasta', 'wt') as opf:
        opf.write('>' + cmd_args.prefix + ' true locations\n')
        opf.write(locs + '\n')

    with open(cmd_args.prefix + '_true_oris.fasta', 'wt') as opf:
        opf.write('>' + cmd_args.prefix + ' true orientations\n')
        opf.write(oris + '\n')

    with open(cmd_args.prefix + '_true_top.fasta', 'wt') as opf:
        opf.write('>' + cmd_args.prefix + ' true topologies\n')
        opf.write(tops + '\n')


if __name__ == '__main__':
    main()