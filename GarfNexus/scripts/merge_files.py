# Notebook to slim the production files for the LPR
import os
import glob
import pandas as pd
import dask.dataframe as dd

# Now load in the stack 
directory_path = '../data/NEXT100_eminus/'

file_paths = glob.glob(os.path.join(directory_path, '*.h5'))

ddf = dd.read_hdf(file_paths, key = 'parts')
parts = ddf.compute()
print(parts)

# Same with the hits
ddf = dd.read_hdf(file_paths, key = 'hits')
hits = ddf.compute()
print(hits)

# Same with the hits
ddf = dd.read_hdf(file_paths, key = 'sns_response')
sns_response = ddf.compute()
print(sns_response)

# Same with the hits
ddf = dd.read_hdf(file_paths, key = 'sns_position')
sns_positions = ddf.compute()
print(sns_positions)

# Open the HDF5 file in write mode
with pd.HDFStore(f"NEXT100_eminus_merged.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    # store.put('config', config, format='table')
    store.put('parts',parts, format='table')
    store.put('hits',hits, format='table')
    store.put('sns_response',sns_response, format='table')
    store.put('sns_positions',sns_positions, format='table')

