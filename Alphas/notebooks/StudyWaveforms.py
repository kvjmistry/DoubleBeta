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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import trim_mean

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


def check_summed_baseline(wfs_sum, grass_lim, S1_height):

    flag=False
    tc=25e-3
    
    # Check the baseline at the end and start match
    # otherwise there could be signal there so the baseline is all messed up
    num_samples = int(25/tc)
    baseline1=np.mean(wfs_sum[-num_samples:])
    baseline2=np.mean(wfs_sum[0:int(25/tc)])

    # Check if the baseline varies by more than 0.1 PE
    if (abs(baseline1-baseline2) > 0.2):
        print("Error in baselines at start and end, dropping event")
        print(baseline1-baseline2)
        flag = True

    # Look in the window for large peaks that could be other S1/S2 pulses. 
    # This will mess up the reconstruction
    peaks, _ = find_peaks(wfs_sum[ int(grass_lim[0]/tc):int(grass_lim[1]/tc)], height=S1_height, distance=30/tc)

    bin_fact=100 # rebin amount
    wfs_sum_binned = rebin_array(wfs_sum, bin_fact)
    peaks_rebin, _ = find_peaks(wfs_sum_binned[ int(grass_lim[0]/(tc*bin_fact)):int(grass_lim[1]/(tc*bin_fact))], height=0.6)
    peaks_rebin = peaks_rebin*bin_fact

    peaks_filt = np.append(peaks, peaks_rebin)

    return flag, peaks_filt

def get_PEs_inWindow(times, wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, window):

    df = []

    tsel_     = in_range(times, *window)
    wfs = wfs[:, tsel_]

    for pmt_no, wf in enumerate(wfs):

        idx_over_thr = np.argwhere(wf > noise[pmt_no]).flatten()
        splits       = np.argwhere(np.diff(idx_over_thr) > thr_split).flatten()
        idx_slices   = np.split(idx_over_thr, splits+1)
        idx_slices   = list(filter(lambda sl: in_range(len(sl), peak_minlen, peak_maxlen + .5), idx_slices))

        for sl in idx_slices:
            m = np.argmax(wf[sl]) + sl[0]
            pe_int = wf[m-half_window:m+half_window].sum()*tc
            df.append(pd.DataFrame(dict(event = evt_info[evt_no][0], ts_raw=ts/1e3, pmt=pmt_no, pe_int=pe_int, peak_time=m*tc+grass_lim[0], noise_thr=noise[pmt_no]), index=[0]))

    return df


def CorrectRawBaseline(wfs):

    corrected_waveforms = []

    for wfm in wfs:
        num_samples = int(25/tc)
        baseline1=np.mean(wfm[-num_samples:])
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
        if (conv_factors[w] == 1e5):
             wfs[w] = wfs[w]*0
        else:
            wfs[w] = wfs[w]/conv_factors[w]

    return wfs

def find_highest_sipm(wfs, event_number):
    assert len(wfs.shape) == 3, "input must be 3-dimensional"

    wfs   = wfs[event_number]
    index = np.argmax(np.max(wfs, axis=1))
    print(index)
    return index

# Get interpolation of the background noise just after the S2
def GetBaselineInterp(times, wfs):

    interps = []
    for pmt_no, wf in enumerate(wfs):

        # Apply trimmed mean over a sliding window
        window_size = 100  # Adjust as needed
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


# Uses the interpolation in a range to fix the baseline
def CorrectDeconvBaseline(t_pmt, tmin, tmax, interps, wfs):
    
    for pmt_no, wf in enumerate(wfs):
        # Apply the interpolation function to waveform time points
        amplitude_interpolated = interps[pmt_no](t_pmt)
        mask = (t_pmt < tmin) | (t_pmt > tmax)
        amplitude_interpolated[mask] = 0

        # Subtract the interpolated amplitude from the waveform amplitude
        wfs[pmt_no] = wf - amplitude_interpolated

    return wfs


# Fit the distribution 
def Gaussian(x, A, mu, sigma):
    return A * np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

def FitS2(time, amplitude):
    max_amplitude = np.max(amplitude)
    half_max = max_amplitude / 2

    # Find indices where amplitude crosses half-maximum level
    above_half_max = np.where(amplitude >= half_max)[0]

    time_slice = time[above_half_max]

    # Perform the curve fit
    params, covariance = curve_fit(Gaussian, time[above_half_max], amplitude[above_half_max], bounds = ([1, min(time_slice),1], [1e6, max(time_slice),100]))

    # Extract the fitted parameters
    A, mu, sigma = params

    return A, mu, sigma

# Function to get the mean and standard deviation of the baseline correction at grass lim start
def GetInterpParams(interps, grass_lim):
    interp_amp = np.array([])
    for interp in interps:
        interp_amp = np.append(interp_amp, interp(grass_lim[0]))

    mean_interp_amp=np.mean(interp_amp)
    std_interp_amp = np.std(interp_amp)

    return mean_interp_amp, std_interp_amp

# Rebin a waveform
def rebin_array(arr, bin_size):
    new_len = len(arr) // bin_size  # Ensure integer division
    return np.mean(arr[:new_len * bin_size].reshape(new_len, bin_size), axis=1)


filename  = sys.argv[1]
RUN_NUMBER= int(sys.argv[2])
base_name = os.path.basename(filename)  # Extracts 'run_13852_0000_ldc1_trg0.waveforms.h5'
outfilename = base_name.replace(".waveforms", "_filtered")

useRaw = False

thr_split   = 2      # maximum number of samples allowed to be below threshold to consider it a peak
peak_minlen = 2      # minimum number of samples above threshold in a peak
peak_maxlen = 10     # maximum number of samples above threshold in a peak
half_window = 4      # number of samples to each side of a peak maximum to integrate
tc          = 25e-3  # constant to convert from samples to time or vice versa. 
noise_sigma = 4      # how many STD above noise for the single PEs to be

dead_pmts = [3, 16, 24, 36, 37, 38, 39,40, 41, 48, 53, 58]

wf_sum = 0

if (RUN_NUMBER == 13850):
    grass_lim   = 1050, 1770 # time window in mus in which to search for single pes
    noise_lim   = 1900, 2000 # time window to calculate the noise baseline
    S1_height   = 10
    S2_height   = 10
    S2_start    = 990        # S2 integration window start
    S2_end      = 1040       # S2 integration window end
    cath_lim    = 1785, 1860 # start/end window for cathode events
    S1_window   = 100, 985   # window to search for S1
    S2_window   = 985, 1200  # window to search for S2

elif (RUN_NUMBER == 13859):
    grass_lim   = 1300, 1770 # time window in mus in which to search for single pes
    noise_lim   = 1900, 2000 # time window to calculate the noise baseline
    S1_height   = 10
    S2_height   = 10
    S2_start    = 990        # S2 integration window start
    S2_end      = 1040       # S2 integration window end
    cath_lim    = 1785, 1860 # start/end window for cathode events
    S1_window   = 100, 985   # window to search for S1
    S2_window   = 985, 1200  # window to search for S2

elif (RUN_NUMBER == 14180):
    grass_lim   = 1050, 1770 
    noise_lim   = 1900, 2000 
    S1_height   = 10
    S2_height   = 10
    S2_start    = 990
    S2_end      = 1040
    cath_lim    = 1785, 1860
    S1_window   = 100, 985
    S2_window   = 985, 1200 

elif (RUN_NUMBER == 14498):
    grass_lim   = 1650, 2350
    noise_lim   = 2500, 2600
    S1_height   = 4
    S2_height   = 8
    S2_start    = 1590
    S2_end      = 1640
    cath_lim    = 2500, 2550 
    S1_window   = 100, 1585
    S2_window   = 1585, 1800 

elif (RUN_NUMBER == 14780): 
    grass_lim   = 1650, 2350
    noise_lim   = 2500, 2600
    S1_height   = 4
    S2_height   = 8
    S2_start    = 1590
    S2_end      = 1640
    cath_lim    = 2500, 2600
    S1_window   = 100, 1585
    S2_window   = 1585, 1800 

else:
    print("No run found, using default argon values...")
    grass_lim   = 1050, 1770 
    noise_lim   = 1900, 2000 
    S1_height   = 10000
    S2_height   = 50000
    S2_start    = 990
    S2_end      = 1040
    cath_lim    = 1785, 1860
    S1_window   = 100, 985
    S2_window   = 985, 1200 


deconv = deconv_pmt("next100", RUN_NUMBER, 62400)

# Load in the database for SiPMs
detector_db = "next100"
datasipm = load_db.DataSiPM(detector_db, RUN_NUMBER)
datapmt = load_db.DataPMT(detector_db, RUN_NUMBER)

data = []
data_properties = []
noise_df = []

with tb.open_file(filename) as file:
    evt_info = file.root.Run.events
    rwf      = file.root.RD.pmtrwf
    time     = np.arange(rwf.shape[2]) * tc
    # tsel     = in_range(time, *grass_lim)
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

        # Zero out the dead PMTs
        if (RUN_NUMBER != 14780):
            for pmt_ in dead_pmts:
                wfs[pmt_] = np.arange(wfs[pmt_] .size) * 0
        
        wfs_sum = sum_wf(wfs)

        times   = np.arange(wfs_sum .size) * tc # sampling period in mus

        S1, _ = find_peaks(wfs_sum[ int(S1_window[0]/tc):int(S1_window[1]/tc)], height=S1_height, distance=10/tc)
        S2, _ = find_peaks(wfs_sum[ int(S2_window[0]/tc):int(S2_window[1]/tc)], height=S2_height, distance=50/tc)

        if (len(S1) ==0):
            print("No S1!")

        if (len(S2) ==0):
            print("No S2!")

        if (len(S1) !=1 or len(S2)!=1 ):
            deltaT = -999
        else:
            deltaT = S2[0]*tc+S2_window[0] - (S1[0]*tc+S1_window[0])

        # Calculate the noise of the PMT
        noise = []
        for pmt_no, wf in enumerate(wfs):
            noise.append(noise_sigma*np.std(wf[int(noise_lim[0]/tc):int(noise_lim[1]/tc)]))

        interps = GetBaselineInterp(times, wfs)

        mean_interp_amp, std_interp_amp = GetInterpParams(interps, grass_lim)

        # Correct the waveforms
        wfs = CorrectDeconvBaseline(times, grass_lim[0], noise_lim[1], interps, wfs) # Correct baseline to the end of the waveform

        wfs_sum_cor = sum_wf(wfs)

        # Check if the corrected event failed the quality control
        pass_flag, grass_peaks = check_summed_baseline(wfs_sum_cor, grass_lim, S1_height)
        if (pass_flag):
            print("Event Failed Quality Control...")
            continue

        if (len(grass_peaks!=0)):
            print("Num Grass Peaks:", len(grass_peaks))

        # Sum values in the peak up to the point where the pulse goes to zero
        S2_area = wfs_sum[int(S2_start/tc):int(S2_end/tc)]
        S2_area = S2_area[S2_area > 0].sum()*tc

        cath_df   = get_PEs_inWindow(times, wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, cath_lim)
        cath_df   = pd.concat(cath_df, ignore_index=True)
        cath_area = cath_df.pe_int.sum()
        cath_time = cath_df.peak_time.mean()
        cath_std  = cath_df.peak_time.std()

        try:
            FWHM, S2_amplitude = find_fwhm(times[int(S2_start/tc):int(S2_end/tc)], 
                                        wfs_sum[int(S2_start/tc):int(S2_end/tc)])
        except:
            print("Error on FWHM calculation, skipping")
            continue # Continue without interruption

        # Also try fitting
        try:
            A, mu, sigma = FitS2(times[int(S2_start/tc):int(S2_end/tc)], wfs_sum[int(S2_start/tc):int(S2_end/tc)])
            area = A * sigma * np.sqrt(2 * np.pi)
        except:
            print("Error on FWHM calculation, skipping")
            continue # Continue without interruption

        data_properties.append(pd.DataFrame(dict(event = evt_info[evt_no][0], S2_area=S2_area, S2_areafit=area, S2_time = mu, cath_area=cath_area, cath_time=cath_time, cath_std=cath_std, ts_raw=ts/1e3, deltaT=deltaT, sigma = FWHM/2.355, S2_amp=S2_amplitude, x = x_pos, y = y_pos, grass_peaks = len(grass_peaks), nS1 = len(S1), mean_interp_amp= mean_interp_amp, std_interp_amp=std_interp_amp), index=[0]))

        # Check the baseline, if we got something really negative
        # then the deconvolution likely failed, so skip grass calculation
        min_baseline = min(wfs_sum_cor[int((grass_lim[0]-50)/tc):int((grass_lim[0]+100)/tc)])
        if (min_baseline > -5):
            df = get_PEs_inWindow(times, wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, grass_lim)
            data = data + df

        else:
            print("Problem with deconvolution", min_baseline)

        # now add a search window before the S1 to understand the background
        if (len(S1)== 0):
            noise_range = 0,500
        else:
            noise_range = 0, S1[0]*tc+S1_window[0]-10
        print("Noise range is:", noise_range[1])
        
        noise_df_temp = get_PEs_inWindow(times, wfs, noise, thr_split, peak_minlen, peak_maxlen, half_window, noise_range)
        noise_df = noise_df + noise_df_temp

    data = pd.concat(data, ignore_index=True)
    data = data.assign(ts = np.array(list(map(datetime.fromtimestamp, data.ts_raw))))
    noise_df = pd.concat(noise_df, ignore_index=True)

    data_properties = pd.concat(data_properties, ignore_index=True)
    data_properties = data_properties.assign(ts = np.array(list(map(datetime.fromtimestamp, data_properties.ts_raw))))

# Merge the noise
grouped_noise = noise_df.groupby(['event'])['pe_int'].sum().reset_index()
grouped_noise = grouped_noise.rename(columns={'pe_int': 'bkg'})

data = data.merge(grouped_noise, on='event', how='left')
data_properties = data_properties.merge(grouped_noise, on='event', how='left')

print(data_properties)
print(data)

with pd.HDFStore(f"/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/filtered/{RUN_NUMBER}/"+outfilename, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', data, format='table')
    store.put('data_properties', data_properties, format='table')
