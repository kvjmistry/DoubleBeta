# Script to merge the h5 files into one

import os
import glob
import pandas as pd


directory_path = '../data/nexus/rebinning/bb/'

file_paths = glob.glob(os.path.join(directory_path, '*'))

df_list = []

for file_path in file_paths:
    print("On file: ", file_path)
    df_list.append(pd.read_hdf(file_path, 'hits'))

df_merge = pd.concat(df_list, ignore_index=True)

print(df_merge)

print("Writing to file...")
with pd.HDFStore(f"../data/nexus/rebinning_merge/xesphere_1bar_bb_merge.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('hits', df_merge, format='table')