# Notebook to slim the filtered raw waveform files
import os
import glob
import pandas as pd
import pickle

# Now load in the stack 
RUN_NUMBER=14180

# mode="filt"
# mode="data"
mode="hist"


if mode == "filt":
    print("Combining Data Properties")
    directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filtered/{RUN_NUMBER}/"
    outfile=f"Run_{RUN_NUMBER}_Filtered.h5"
    
    file_paths = glob.glob(os.path.join(directory_path, '*.h5'))
    print(len(file_paths))

    data_properties = []

    for file_path in file_paths:

        data_properties.append(pd.read_hdf(file_path, key = 'data_properties'))

    data_properties  = pd.concat(data_properties)

    print(data_properties)

    # Open the HDF5 file in write mode
    with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
        store.put('data_properties',data_properties, format='table')

elif (mode == "data"):
    print("Combining Data Table")
    directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filteredC/{RUN_NUMBER}/"
    outfile=f"Run_{RUN_NUMBER}_FilteredC.h5"
    
    file_paths = glob.glob(os.path.join(directory_path, '*.h5'))
    print(len(file_paths))

    data = []

    for file_path in file_paths:
        data.append(pd.read_hdf(file_path, key = 'data'))

    data = pd.concat(data)

    data = data[["event", "pmt", "pe_int", "peak_time", "pe_intC"]]

    print(data)

    # Open the HDF5 file in write mode
    with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
        store.put('data',data, format='table')

else:
    directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filteredC/{RUN_NUMBER}/"
    outfilehist=f"Run_{RUN_NUMBER}_Histograms.pkl"
    
    file_paths = glob.glob(os.path.join(directory_path, '*.pkl'))
    print(len(file_paths))

    histogram_df = []

    total_hist = None

    for file_path in file_paths:
        histogram_df.append(pickle.load(file_path))
        hist2D = pickle.load(file_path)

        if total_hist is None:
            total_hist = hist2D
        else:
            total_hist += hist2D

    histogram_df = pd.concat(histogram_df)

    print(histogram_df)

    with open(outfilehist, 'wb') as pickle_file:
        pickle.dump(histogram_df, pickle_file)
        pickle.dump(total_hist, pickle_file)