import pandas as pd
import numpy as np
import tables as tb
import os
import sys
import glob


def CorrectLifetimeAvg(row, var, t, mean_lt):

    time = row[t]

    if (mean_lt == 0):
        return 1
    else:
        return row[var] / np.exp(-1*time/mean_lt)


filename  = sys.argv[1]
RUN_NUMBER= int(sys.argv[2])
base_name = os.path.basename(filename)  # Extracts 'run_13852_0000_ldc1_trg0.waveforms.h5'
outfilename = base_name.replace("_filtered", "_filteredC")

data = pd.read_hdf(filename, "data")

if (RUN =="14180"):
    mean_lt = 52000.0 # mus
    trig_time = 1000


data['peak_time'] = data['peak_time'] - trig_time # Need to correct to get the right drift time
data["pe_intC"]   = data.apply(lambda row: CorrectLifetimeAvg(row, "pe_int", "peak_time",  mean_lt), axis=1)


with pd.HDFStore(f"/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/filteredC/{RUN_NUMBER}/"+outfilename, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', data, format='table')