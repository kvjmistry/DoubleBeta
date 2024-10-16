from invisible_cities.cities.components import deconv_pmt
from invisible_cities.core.core_functions import in_range
from invisible_cities.database  import load_db
import pandas as pd
import numpy as np
import tables as tb
import os
from scipy.signal import find_peaks
from pathlib  import Path
import sys
from datetime import datetime

def sum_wf(wfs, event_number):
    assert len(wfs.shape) == 3, "input must be 3-dimensional"

    wfs   = wfs[event_number]

    element_wise_sum = np.zeros_like(wfs[0],dtype=np.int64)

    # Sum the arrays element-wise
    for array in wfs:
        element_wise_sum += array

    return element_wise_sum

def sum_wf(wfs):
    element_wise_sum = np.zeros_like(wfs[0])

    # Sum the arrays element-wise
    for array in wfs:
        element_wise_sum += array

    return element_wise_sum


def check_summed_baseline(wfs_sum, grass_lim, scale_factor):

    flag=False
    tc=25e-3
    
    # Check the baseline at the end and start match
    # otherwise there could be signal there so the baseline is all messed up
    baseline1=np.mean(wfs_sum[ int(1975/tc):int(2000/tc)])
    baseline2=np.mean(wfs_sum[0:int(25/tc)])

    # 20 seems like a good number to check the difference against
    if (abs(baseline1-baseline2) > 20*scale_factor):
        print("Error in baselines at start and end, dropping event")
        print(baseline1-baseline2)
        flag = True

    # Look in the window for large peaks that could be other S2 pulses. 
    # This will mess up the reconstruction
    peaks, _ = find_peaks(wfs_sum[ int(grass_lim[0]/tc):int(grass_lim[1]/tc)], height=8000, distance=40/tc)

    return flag, peaks

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


def CorrectRawBaseline(wfs):

    corrected_waveforms = []

    for wfm in wfs:
        baseline1=np.mean(wfm[ int(1975/tc):int(2000/tc)])
        baseline2=np.mean(wfm[0:int(25/tc)])
        wfm = -1*(wfm-baseline2)
        corrected_waveforms.append(wfm)

    return np.array(corrected_waveforms)


def find_fwhm(time, amplitude):
    max_amplitude = np.max(amplitude)
    half_max = max_amplitude / 2

    # Find indices where amplitude crosses half-maximum level
    above_half_max = np.where(amplitude >= half_max)[0]

    # First crossing point
    left_idx = above_half_max[0]
    right_idx = above_half_max[-1]

    # Interpolate to get more accurate crossing times
    t_left = np.interp(half_max, [amplitude[left_idx-1], amplitude[left_idx]], [time[left_idx-1], time[left_idx]])
    t_right = np.interp(half_max, [amplitude[right_idx], amplitude[right_idx+1]], [time[right_idx], time[right_idx+1]])

    fwhm = t_right - t_left
    return fwhm, max_amplitude

def ADC_to_PE(wfs, datapmt):
    conv_factors = datapmt.adc_to_pes.to_numpy()

    for w in range(0, len(wfs)):
        wfs[w] = wfs[w]*conv_factors[w]

    return wfs

def find_highest_sipm(wfs, event_number):
    assert len(wfs.shape) == 3, "input must be 3-dimensional"

    wfs   = wfs[event_number]
    index = np.argmax(np.max(wfs, axis=1))
    print(index)
    return index


filename  = sys.argv[1]
RUN_NUMBER= int(sys.argv[2])
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
scale_factor = 40*60 # Scale factor for summed waveform. 60 Pmts, 40 is ~ the conversion factor

useRaw = True

wf_sum = 0


deconv = deconv_pmt("next100", RUN_NUMBER, 62400)

# Load in the database for SiPMs
detector_db = "next100"
datasipm = load_db.DataSiPM(detector_db, RUN_NUMBER)
datapmt = load_db.DataPMT(detector_db, RUN_NUMBER)

data = []
data_properties = []
with tb.open_file(filename) as file:
    evt_info = file.root.Run.events
    rwf      = file.root.RD.pmtrwf
    time     = np.arange(rwf.shape[2]) * tc
    tsel     = in_range(time, *grass_lim)
    for evt_no, wfs in enumerate(rwf):

        print("On event: ", evt_no, " (", evt_info[evt_no][0], ")")

        highest_sipm = find_highest_sipm(file.root.RD.sipmrwf, evt_no)
        x_pos = datasipm.iloc[highest_sipm].X
        y_pos = datasipm.iloc[highest_sipm].Y

        _, ts = evt_info[evt_no]
        
        if ( useRaw):
            wfs = CorrectRawBaseline(wfs)
        else:
            wfs = deconv(wfs)

        # Convert the ADC to PE
        wfs = ADC_to_PE(wfs, datapmt)

        wfs[16] = np.arange(wfs[16] .size) * 0
        
        wfs_sum = sum_wf(wfs)

        times   = np.arange(wfs_sum .size) * 25e-3 # sampling period in mus

        # Check if  event failed the quality control
        pass_flag, grass_peaks = check_summed_baseline(wfs_sum, grass_lim, scale_factor)
        if (pass_flag):
            print("Event Failed Quality Control...")
            continue

        S1, _ = find_peaks(wfs_sum[ int(100/tc):int(985/tc)], height=10000, distance=40/tc)
        S2, _ = find_peaks(wfs_sum[ int(985/tc):int(1200/tc)], height=200000, distance=200/tc)

        if (len(S1) !=1 or len(S2)!=1 ):
            deltaT = -999
        else:
            deltaT = S2[0]*tc+985 - (S1[0]*tc+100)

        # Calcilate the noise of the PMT
        noise = []
        for pmt_no, wf in enumerate(wfs):
            noise.append(noise_sigma*np.std(wf[int(noise_lim[0]/tc):int(noise_lim[1]/tc)]))


        # Sum values in the peak up to the point where the pulse goes to zero
        S2_area = wfs_sum[int(990/tc):int(1030/tc)]
        S2_area = S2_area[S2_area > 0].sum()

        cath_df = get_PEs_inWindow(wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, [1780,1850])
        cath_df = pd.concat(cath_df, ignore_index=True)
        cath_area = cath_df.pe_int.sum()

        FWHM, S2_amplitude = find_fwhm(times[int(990/tc):int(1030/tc)], wfs_sum[int(990/tc):int(1030/tc)])

        data_properties.append(pd.DataFrame(dict(event = evt_info[evt_no][0], S2_area=S2_area,cath_area=cath_area, ts_raw=ts/1e3, deltaT=deltaT, sigma = FWHM/2.355, S2_amp=S2_amplitude, x = x_pos, y = y_pos, grass_peaks = len(grass_peaks), nS1 = len(S1)), index=[0]))

        df = get_PEs_inWindow(wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, grass_lim)
        data = data + df

    data = pd.concat(data, ignore_index=True)
    data = data.assign(ts = np.array(list(map(datetime.fromtimestamp, data.ts_raw))))

    data_properties = pd.concat(data_properties, ignore_index=True)
    data_properties = data_properties.assign(ts = np.array(list(map(datetime.fromtimestamp, data_properties.ts_raw))))

print(data_properties)

with pd.HDFStore(f"/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/filtered/{RUN_NUMBER}/"+outfilename, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', data, format='table')
    store.put('data_properties', data_properties, format='table')
