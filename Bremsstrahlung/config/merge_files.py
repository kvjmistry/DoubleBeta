# Notebook to slim the production files for the LPR
import os
import glob
import pandas as pd

# Now load in the stack 
MODE="WVI"
directory_path = f"/media/argon/HDD_8tb/Krishan/Bremsstrahlung/{MODE}/*/"
outfile=f"Next100_Tl208_Port1a_{MODE}_slim_merged.h5"
 
file_paths = glob.glob(os.path.join(directory_path, '*.h5'))
print(len(file_paths))

parts = []
hits = []

for file_path in file_paths:
    parts.append(pd.read_hdf(file_path, key = 'MC/particles', columns = ["event_id", "particle_name", "creator_proc", "kin_energy"]))
    hits.append(pd.read_hdf(file_path, key = 'MC/hits',       columns = ["event_id", "x", "y", "z", "energy", "particle_id"]))

parts = pd.concat(parts)
hits  = pd.concat(hits)

print(parts)
print(hits)

# Open the HDF5 file in write mode
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    # store.put('config', config, format='table')
    store.put('MC/particles',parts, format='table')
    store.put('MC/hits',hits, format='table')

