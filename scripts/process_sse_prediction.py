#!/usr/bin/env python3

import sys


def main():
    
    ciphipsi_file = sys.argv[1]
    sse_file = sys.argv[2]
    prefix = sys.argv[3]
    
    # get the set of positions with missing residue
    missing_res_set = set()
    with open(ciphipsi_file, 'rt') as ipf:
        for line in ipf:
            l_stripped = line.strip()
            if l_stripped.startswith('#'):
                continue
            # get targets
            ss = l_stripped.split()[3]
            # skip unresolved residues
            if ss not in ['H', 'E', 'C']:
                missing_res_set.add(int(l_stripped.split()[0]))
                    
    with open(sse_file, 'rt') as ipf:
        header = next(ipf).strip()
        top_str = ''.join([l.strip() for l in ipf])
    
    new_top_str = ''
    for i, r in enumerate(top_str, start=1):
        if i in missing_res_set:
            continue
        else:
            new_top_str += r
            
    with open(prefix + '_processed_sse.fasta', 'wt') as opf:
        opf.write(header)
        opf.write('\n')
        opf.write(new_top_str)
        opf.write('\n')


if __name__ == '__main__':
    main()