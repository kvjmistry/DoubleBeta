from invisible_cities.cities.components import wf_from_files, WfType
from invisible_cities.cities.components import deconv_pmt, calibrate_pmts, get_pmt_wfs, load_dst
import pandas as pd
import numpy as np
import tables as tb
import os


kdst_path = "../data/DEMO/kdst/"
raw_path  = "../data/DEMO/raw/"

kdst_path = "/media/argon/HDD_8tb/Krishan/DEMO/kdst/"
raw_path  = "/media/argon/HDD_8tb/Krishan/DEMO/raw/"

file_list = os.listdir(kdst_path)

evt_filter_kdst = []
kdst_dfs = []

for file in file_list:

    path = kdst_path + file
    kdst = pd.read_hdf(path, "/DST/Events")
    # kdst = kdst[kdst.nS1 == 1]
    kdst = kdst[kdst.nS2 > 1]
    # kdst = kdst[kdst.Z < 25]
    kdst = kdst[  np.sqrt(kdst.X*kdst.X + kdst.Y*kdst.Y) < 60]

    # This removes events where the zrms is not consistent with the z value
    # kdst = kdst[ (kdst.Zrms > 0.0035*kdst.Z+0.75) & (kdst.Zrms < 0.0035*kdst.Z+0.95)  ]

    # display(kdst)

    evt_filter_kdst.append(kdst.event.unique())
    kdst_dfs.append(kdst)

evt_filter_kdst = np.concatenate(evt_filter_kdst, axis=0)
kdst_dfs = pd.concat(kdst_dfs)
print(kdst_dfs)

file_list_raw = os.listdir(raw_path)

rw_dfs = []

for file in file_list_raw:

    path = raw_path + file

    # raw_evts = pd.read_hdf(rawfile, '/Run/events')
    raw_evts = load_dst(path, 'Run', 'events')
    # display(raw_evts)
    nrwfs = len(raw_evts.evt_number.unique())
    print("Number of Waveforms:", nrwfs)

    raw_wfs = wf_from_files([path], WfType.rwf)
    

    # Loop over waceforms and display them
    for rwf, irwf, evt_no in zip(raw_wfs, range(nrwfs), raw_evts.evt_number):
        
        evt_filter_kdst
        if evt_no in evt_filter_kdst:

            z = kdst_dfs[kdst_dfs.event == evt_no].Z.iloc[0]

            # Deconvolve the PMT Waveforms
            cwf = deconv_pmt("demopp", 12081, 35000)(rwf['pmt'])
            
            # Calibrate the deconvolved PMT Waveforms
            ccwfs, ccwfs_mau, cwf_sum, cwf_sum_mau = calibrate_pmts("demopp", 12081, 100, 3)(cwf)

            times = np.array([i*25/1000 for i in range(len(np.sum(rwf['pmt'], axis=0)))])
            
            # Sum the waveforms
            sumwf = np.sum(ccwfs, axis=0)

            df = pd.DataFrame({"times" : times, "sumwf" : sumwf})
            df["z"] = z
            df["event"] = evt_no

            df = df[df.times > 385]
            df = df[df.times < 425]
    
            # display(df_merged)

            rw_dfs.append(df)

rw_dfs_merge = pd.concat(rw_dfs)

with pd.HDFStore("DEMO_Slim.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('wfm', rw_dfs_merge, format='table')
    store.put('kdst', kdst_dfs, format='table')