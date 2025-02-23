import pandas as pd
import numpy as np
import tables as tb
from scipy.stats import trim_mean
from scipy.interpolate import interp1d
from invisible_cities.database  import load_db
import sys
import os


def GetBaselineInterp(times, wfs):

    interps = []
    for pmt_no, wf in enumerate(wfs):

        # Apply trimmed mean over a sliding window
        window_size = 20  # Adjust as needed
        trim_ratio = 0.4  # Trim 10% from each side

        trimmed_wfm = np.array([
            trim_mean(wf[i:i+window_size], proportiontocut=trim_ratio) 
            for i in range(0, len(wf), window_size)
        ])

        # Downsample the time array to match the trimmed values
        trimmed_times = times[::window_size]

        # Interpolate the smoothed data
        interp_func = interp1d(trimmed_times, trimmed_wfm, kind='cubic', bounds_error=False, fill_value=0)  # Cubic interpolation for smoothness
        interps.append(interp_func)

    return interps


def CorrectRawBaseline(wfs):

    corrected_waveforms = []

    for wfm in wfs:
        num_samples = int(25)
        baseline2=np.mean(wfm[0:int(25)])
        wfm = wfm-baseline2
        corrected_waveforms.append(wfm)

    return np.array(corrected_waveforms)


def GetS2Areas(wfs):

    S2_areas = []
    S2_start    = 1590
    S2_end      = 1640

    for wf_sipm in wfs:
        S2_area = wf_sipm[int(S2_start):int(S2_end)]
        S2_area = S2_area[S2_area > 0].sum()
        S2_areas.append(S2_area)

    return S2_areas
    

def BaselineFit(interps, t_start, t_end):

    new_time = np.linspace(t_start,t_end,500)

    slopes = []
    intercepts = []
    offsets = []
    for interp in interps:
        interpolated_amplitude = interp(new_time)

        slope, intercept = np.polyfit(new_time, interpolated_amplitude, 1)
        offset = slope * t_start + intercept

        slopes.append(slope)
        intercepts.append(intercept)
        offsets.append(offset)

    return slopes,intercepts, offsets


filename  = sys.argv[1]
RUN_NUMBER= int(sys.argv[2])
base_name = os.path.basename(filename)  # Extracts 'run_13852_0000_ldc1_trg0.waveforms.h5'
outfilename = base_name.replace(".waveforms", "_sipm")


detector_db = "next100"
datasipm = load_db.DataSiPM(detector_db, RUN_NUMBER)

baselinewindow =  1700, 3000

sipm_properties = []

with tb.open_file(filename) as file:

    evt_info = file.root.Run.events
    rwf      = file.root.RD.sipmrwf
    
    # tsel     = in_range(time, *grass_lim)
    for evt_no, wfs in enumerate(rwf):

        print("On Event:", evt_no)

        wfs = rwf[evt_no]

        # # This corrects the baseline from ~50 to zero by shifting it
        wfs = CorrectRawBaseline(wfs)
        
        sp_sipm = 1     # sampling period in mus
        times  = np.arange(wfs[0].size) * sp_sipm

        S2_areas = GetS2Areas(wfs)

        interps = GetBaselineInterp(times, wfs)
        slopes, intercepts, offsets = BaselineFit(interps, baselinewindow[0],baselinewindow[1])

        ## add to the dataframe
        for index in datasipm.index:
            if (S2_areas[index] < 1000):
                continue
            sipm_properties.append(pd.DataFrame(dict(event = evt_no, sipm_index=index,S2_area=S2_areas[index], X=datasipm.iloc[index].X, Y=datasipm.iloc[index].Y, slope=slopes[index], intercept=intercepts[index], offset=offsets[index]), index=[0]))

sipm_properties = pd.concat(sipm_properties, ignore_index=True)

with pd.HDFStore(f"/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/sipm_baselines/{RUN_NUMBER}/"+outfilename, mode='w', complevel=5, complib='zlib') as store:
    store.put('sipm_properties', sipm_properties, format='table')