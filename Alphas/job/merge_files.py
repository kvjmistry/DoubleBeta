# Notebook to slim the filtered raw waveform files
import os
import glob
import pandas as pd
import pickle

# Now load in the stack 
RUN_NUMBER=13850
# RUN_NUMBER=13859
# RUN_NUMBER=14180
# RUN_NUMBER=14498
# RUN_NUMBER=14780

mode="filt"
# mode="data"
# mode="hist"
# mode="sipm"


if mode == "filt":
    print("Combining Data Properties")
    directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filtered/{RUN_NUMBER}/"
    outfile=f"files/Run_{RUN_NUMBER}_Filtered.h5"
    
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
    outfile=f"files/Run_{RUN_NUMBER}_FilteredC.h5"
    
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

elif (mode == "sipm"):
    print("Combining sipm Table")
    directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/sipm_baselines/{RUN_NUMBER}/"
    outfile=f"files/Run_{RUN_NUMBER}_SiPM.h5"
    
    file_paths = glob.glob(os.path.join(directory_path, '*.h5'))
    print(len(file_paths))

    data = []

    for file_path in file_paths:
        data.append(pd.read_hdf(file_path, key = 'sipm_properties'))

    data = pd.concat(data)

    print(data)

    # Open the HDF5 file in write mode
    with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
        store.put('sipm_properties',data, format='table')


else:
    directory_path = f"/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filteredC/{RUN_NUMBER}/"
    outfilehist=f"files/Run_{RUN_NUMBER}_Histograms.pkl"
    
    file_paths = glob.glob(os.path.join(directory_path, '*.pkl'))
    print(len(file_paths))

    histogram_df = []
    histogram_df1D = []

    total_hist = None

    for file_path in file_paths:
        with open(file_path, 'rb') as pickle_file:
            histogram_df.append(pickle.load(pickle_file))
            histogram_df1D.append(pickle.load(pickle_file))
            hist2D = pickle.load(pickle_file)

            if total_hist is None:
                total_hist = hist2D
            else:
                total_hist += hist2D

    histogram_df = pd.concat(histogram_df)
    histogram_df1D = pd.concat(histogram_df1D)

    print(histogram_df)

    with open(outfilehist, 'wb') as pickle_file:
        pickle.dump(histogram_df, pickle_file)
        pickle.dump(histogram_df1D, pickle_file)
        pickle.dump(total_hist, pickle_file)