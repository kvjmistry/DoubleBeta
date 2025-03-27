import pandas as pd
import numpy as np
import tables as tb
import os
import sys
import glob
import pickle


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
outfilenamepkl = base_name.replace("_filtered.h5", "_pickle.pkl")

print("RUN is", RUN_NUMBER, " file is", base_name)

data = pd.read_hdf(filename, "data")


if (RUN_NUMBER == 14180):
    mean_lt = 43000.0 # mus
    trig_time = 1009
    cathode_time = 803
    PE_to_MeV = 4.1107552268907513e-07
    max_lifetime = 100e3
    bin_range = 4000

elif (RUN_NUMBER == 13850):
    mean_lt = 44000.0 # mus
    trig_time = 1009
    cathode_time = 801
    PE_to_MeV = 4.380634467274111e-07
    max_lifetime = 100e3
    bin_range = 4000

elif (RUN_NUMBER == 13859):
    mean_lt = 5800.0 # mus
    trig_time = 1009
    cathode_time = 760
    PE_to_MeV = 5.80816646495761e-07
    max_lifetime = 100e3
    bin_range = 25000

elif (RUN_NUMBER == 14498):
    mean_lt = 45000.0 # mus
    trig_time = 1610
    cathode_time = 1500
    PE_to_MeV = 3.139451326134709e-07
    max_lifetime = 100e3
    bin_range = 500

elif (RUN_NUMBER == 14780):
    mean_lt = 60000.0 # mus
    trig_time = 1610
    cathode_time = 1500
    PE_to_MeV = 3.0007670255073243e-07
    max_lifetime = 200e3
    bin_range = 500

else:
    print("Error run config is not set")

data['peak_time'] = data['peak_time'] - trig_time # Need to correct to get the right drift time
data["pe_intC"]   = data.apply(lambda row: CorrectLifetimeAvg(row, "pe_int", "peak_time",  mean_lt), axis=1)

# Convert from PE to eV
data["pe_intC"] = data["pe_intC"]*PE_to_MeV*1e6

print(data)

# ---------------------------------------------------------------------------------------------------------------------------------------

# Load the data
data_properties_lt = pd.read_hdf(f"/home/argon/Projects/Krishan/DoubleBeta/Alphas/CalibratedProperties/Properties_Run_{RUN_NUMBER}.h5", "data_properties_lt")

bins = np.arange(50, 750, 50)
bin_centers = (bins[:-1] + bins[1:]) / 2

total_hist = None

tail_energy = []
S2_areas = []
events = []
x_binc = []
y_binc = []
bin_ids = []

histogram1D_df = []

for index, evt in enumerate(data.event.unique()):

    print(f"Event: {index}, {evt}")

    S2_pulse = data_properties_lt[data_properties_lt['event'] == evt]

    if (len(S2_pulse) ==0):
        continue

    S2_area = S2_pulse.S2_areaC.item()
    grass_peaks = S2_pulse.grass_peaks.item()
    event = data[data.event == evt]

    if (S2_area < 0.5):
        continue

    if (grass_peaks >0):
        continue

    counts, edges = np.histogram(event.peak_time, weights=event.pe_intC/S2_area, bins = bins )

    hist2D, xedges, yedges = np.histogram2d(bin_centers, counts, bins=[bins, np.linspace(0, bin_range, 50)])
    
    # masked_hist=hist2D

    if total_hist is None:
        total_hist = hist2D
    else:
        total_hist += hist2D

    tail_energy.append(event.pe_intC.sum())
    S2_areas.append(S2_area)
    events.append(evt)
    x_binc.append( S2_pulse.x_bin_center.item())
    y_binc.append( S2_pulse.y_bin_center.item())
    bin_ids.append( S2_pulse.bin_id.item())
    histogram1D_df.append(pd.DataFrame({"event":evt, "counts":counts, "centers":bin_centers}))


histogram_df = pd.DataFrame({"event":events,"S2_areas":S2_areas,"tail_energy": tail_energy, "x_binc": x_binc, "y_binc" : y_binc, "bin_id" : bin_ids} )
print(histogram_df)

histogram1D_df = pd.concat(histogram1D_df, ignore_index=True)
print(histogram1D_df)

with open(f"/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/filteredC/{RUN_NUMBER}/"+outfilenamepkl, 'wb') as pickle_file:
    pickle.dump(histogram_df, pickle_file)
    pickle.dump(histogram1D_df, pickle_file)
    pickle.dump(total_hist, pickle_file)

# ---------------------------------------------------------------------------------------------------------------------------------------

with pd.HDFStore(f"/media/argon/HDD_8tb/Krishan/NEXT100Data/alpha/filteredC/{RUN_NUMBER}/"+outfilename, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', data, format='table')