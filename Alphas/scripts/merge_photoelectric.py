import glob
import pandas as pd
import pickle
import sys

# Example
# python merge_outputs.py /home/argon/Projects/Krishan/DoubleBeta/Alphas/photoelectric/ /home/argon/Projects/Krishan/DoubleBeta/Alphas/photoelectric/photoelectric

file_path = sys.argv[1]

file_out = sys.argv[2]

files = sorted(glob.glob(f"{file_path}/*/*.h5"))

dfs = []

for f in files:
    print(f)
    parts = pd.read_hdf(f, "MC/particles")
    dfs.append(parts)

dfs = pd.concat(dfs)

print(dfs)

with pd.HDFStore(f"{file_out}_merged.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/particles', dfs, format='table')


for f in $(ls /home/argon/Projects/Krishan/DoubleBeta/Alphas/photoelectric/); do cat ${f}/photoelectric.txt >> ${f}/photoelectric_merged.txt; done
