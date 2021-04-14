import sys


ciphipsi_file = sys.argv[1]
loc_file = sys.argv[2]
prefix = sys.argv[3]

# get the set of positions with missing residue
missing_res_set = set()
with open(ciphipsi_file, 'rt') as ipf:
    for line in ipf:
        l_stripped = line.strip()
        if l_stripped.startswith('#'):
            continue
        # get targets
        ss = l_stripped.split()[4]
        # skip unresolved residues
        if ss not in ['M', 'S']:
            missing_res_set.add(int(l_stripped.split()[0]))
                
with open(loc_file, 'rt') as ipf:
    header = next(ipf).strip()
    loc_str = ''.join([l.strip() for l in ipf])

new_loc_str = ''
for i, r in enumerate(loc_str, start=1):
    if i in missing_res_set:
        continue
    else:
        new_loc_str += r
        
with open(prefix + '_processed_loc.fasta', 'wt') as opf:
    opf.write(header)
    opf.write('\n')
    opf.write(new_loc_str)
    opf.write('\n')
