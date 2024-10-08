# Notebook to slim the filtered raw waveform files
import os
import glob
import pandas as pd

# Now load in the stack 
directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filtered/13850/"
outfile=f"Run_13850_FilteredRaw.h5"
 
file_paths = glob.glob(os.path.join(directory_path, '*.h5'))
print(len(file_paths))

data = []
data_properties = []

for file_path in file_paths:
    data.append(pd.read_hdf(file_path, key = 'data'))
    data_properties.append(pd.read_hdf(file_path, key = 'data_properties'))

data = pd.concat(data)
data_properties  = pd.concat(data_properties)

print(data)
print(data_properties)

# Open the HDF5 file in write mode
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    store.put('data',data, format='table')
    store.put('data_properties',data_properties, format='table')