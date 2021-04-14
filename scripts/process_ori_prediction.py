import sys


ciphipsi_file = sys.argv[1]
ori_file = sys.argv[2]
prefix = sys.argv[3]

# get the set of positions with missing residue
missing_res_set = set()
with open(ciphipsi_file, 'rt') as ipf:
    for line in ipf:
        l_stripped = line.strip()
        if l_stripped.startswith('#'):
            continue
        # get targets
        ss = l_stripped.split()[5]
        # skip unresolved residues
        if ss not in ['L', 'P', '-']:
            missing_res_set.add(int(l_stripped.split()[0]))
                
with open(ori_file, 'rt') as ipf:
    header = next(ipf).strip()
    ori_str = ''.join([l.strip() for l in ipf])

new_ori_str = ''
for i, r in enumerate(ori_str, start=1):
    if i in missing_res_set:
        continue
    else:
        new_ori_str += r
        
with open(prefix + '_processed_ori.fasta', 'wt') as opf:
    opf.write(header)
    opf.write('\n')
    opf.write(new_ori_str)
    opf.write('\n')
