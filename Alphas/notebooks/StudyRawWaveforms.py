from invisible_cities.cities.components import deconv_pmt
from invisible_cities.core.core_functions import in_range
import pandas as pd
import numpy as np
import tables as tb
import os
from scipy.signal import find_peaks
from pathlib  import Path
import sys
from datetime import datetime

def check_summed_baseline(wfs, grass_lim):

    flag=False
    tc=25e-3
    
    wfs_sum = sum_wf(wfs)
    
    # Check the baseline at the end and start match
    # otherwise there could be signal there so the baseline is all messed up
    baseline1=np.mean(wfs_sum[ int(1975/tc):int(2000/tc)])
    baseline2=np.mean(wfs_sum[0:int(25/tc)])

    # 20 seems like a good number to check the difference against
    if (abs(baseline1-baseline2) > 20):
        print("Error in baselines at start and end, dropping event")
        print(baseline1-baseline2)
        flag = True

    # Look in the window for large peaks that could be other S2 pulses. 
    # This will mess up the reconstruction
    peaks, _ = find_peaks(wfs_sum[ int(grass_lim[0]/tc):int(grass_lim[1]/tc)], height=100, distance=40/tc)

    if (len(peaks) > 1):
        flag = True

    return flag

def get_PEs_inWindow(wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, grass_lim):

    df = []

    wfs = wfs[:, tsel]

    for pmt_no, wf in enumerate(wfs):

        idx_over_thr = np.argwhere(wf > noise[pmt_no]).flatten()
        splits       = np.argwhere(np.diff(idx_over_thr) > thr_split).flatten()
        idx_slices   = np.split(idx_over_thr, splits+1)
        idx_slices   = list(filter(lambda sl: in_range(len(sl), peak_minlen, peak_maxlen + .5), idx_slices))

        for sl in idx_slices:
            m = np.argmax(wf[sl]) + sl[0]
            pe_int = wf[m-half_window:m+half_window].sum()
            df.append(pd.DataFrame(dict(event = evt_info[evt_no][0], ts_raw=ts/1e3, pmt=pmt_no, pe_int=pe_int, peak_time=m*tc+grass_lim[0], noise_thr=noise[pmt_no]), index=[0]))

    return df



filename  = sys.argv[1]
base_name = os.path.basename(filename)  # Extracts 'run_13852_0000_ldc1_trg0.waveforms.h5'
outfilename = base_name.replace(".waveforms", "_filtered")

grass_lim   = 1350, 1770 # time window in mus in which to search for single pes
noise_lim   = 1900, 2000 # time window to calculate the noise baseline
thr_1pe     = 3.5 # threshold in adc to consider a peak a 1pe candidate
thr_split   = 2 # maximum number of samples allowed to be below threshold to consider it a peak
peak_minlen = 2  # minimum number of samples above threshold in a peak
peak_maxlen = 10 # maximum number of samples above threshold in a peak
half_window = 4 # number of samples to each side of a peak maximum to integrate
n_dark      = 10 # max number of samples without pe
tc          = 25e-3 # constant to convert from samples to time or vice versa. 
noise_sigma = 4 # how many STD above noise for the single PEs to be

wf_sum = 0


deconv = deconv_pmt("next100", 13850, 62400)

data = []
data_properties = []
with tb.open_file(filename) as file:
    evt_info = file.root.Run.events
    rwf      = file.root.RD.pmtrwf
    time     = np.arange(rwf.shape[2]) * tc
    tsel     = in_range(time, *grass_lim)
    for evt_no, wfs in enumerate(rwf):

        _, ts = evt_info[evt_no]
        wfs = deconv(wfs)
        wfs_sum = sum_wf(wfs)

        # Check if  event failed the quality control
        pass_flag = check_summed_baseline(wfs, grass_lim)
        if (pass_flag):
            print("Skipping event...")
            continue


        # Calcilate the noise of the PMT
        noise = []
        for pmt_no, wf in enumerate(wfs):
            noise.append(noise_sigma*np.std(wf[int(noise_lim[0]/tc):int(noise_lim[1]/tc)]))


        S2_area  = wfs_sum[int(997/tc):int(1030/tc)].sum()
        cath_df = get_PEs_inWindow(wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, [1770,1830])
        cath_df = pd.concat(cath_df, ignore_index=True)
        cath_area = cath_df.pe_int.sum()

        data_properties.append(pd.DataFrame(dict(event = evt_info[evt_no][0], S2_area=S2_area,cath_area=cath_area, ts_raw=ts/1e3), index=[0]))

        df = get_PEs_inWindow(wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, grass_lim)
        data = data + df

    data = pd.concat(data, ignore_index=True)
    data = data.assign(ts = np.array(list(map(datetime.fromtimestamp, data.ts_raw))))

    data_properties = pd.concat(data_properties, ignore_index=True)
    data_properties = data_properties.assign(ts = np.array(list(map(datetime.fromtimestamp, data_properties.ts_raw))))

with pd.HDFStore("/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/filtered/13850/"+outfilename, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', data, format='table')
    store.put('data_properties', data_properties, format='table')
