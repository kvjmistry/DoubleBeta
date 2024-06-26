import os
import pandas as pd

# Define the directory containing the HDF5 files
directory = '/media/argon/HDD_8tb/Krishan/DEMO/nexus/Kr/'

# Initialize an empty list to store DataFrames
parts = []
hits = []
configs = []
sns_responses = []
sns_positions = []
waveforms = []

# Iterate over the files in the directory
for i in range(1, 309):  # Range 1-308
    file_path = os.path.join(directory, f'job{i}/DEMOpp_Kr_Z{i}mm_slim.h5')
    if os.path.exists(file_path):
        # Load tables from the HDF5 file into DataFrames
        config_df = pd.read_hdf(file_path,'config')
        parts_df = pd.read_hdf(file_path,'parts')
        hits_df = pd.read_hdf(file_path,'hits')
        sns_response_df = pd.read_hdf(file_path,'sns_response')
        sns_position_df = pd.read_hdf(file_path,'sns_position')
        # Append DataFrames to the list
        configs.append(config_df)
        parts.append(parts_df)
        hits.append(hits_df)
        sns_responses.append(sns_response_df)
        sns_positions.append(sns_position_df)

        sns_positions_pmt = sns_position_df[sns_position_df.sensor_name == "PmtR11410"]
        merged_df_pmt = pd.merge(sns_response_df, sns_positions_pmt, on='sensor_id', how='right')
        # Convert to time bins in mus
        merged_df_pmt.time_bin = merged_df_pmt.time_bin*25*1e-3
        waveforms.append(merged_df_pmt)


# Merge all DataFrames into a single DataFrame
config_df = pd.concat(configs, ignore_index=True)
print(config_df)

parts_df = pd.concat(parts, ignore_index=True)
print(parts_df)

hits_df = pd.concat(hits, ignore_index=True)
print(hits_df)

sns_response_df = pd.concat(sns_responses, ignore_index=True)
print(sns_response_df)

sns_position_df = pd.concat(sns_positions, ignore_index=True)
print(sns_position_df)

waveform_df = pd.concat(waveforms, ignore_index=True)
print(waveform_df)


# Now you have a single DataFrame containing data from all HDF5 files

with pd.HDFStore("DEMOpp_Kr_garfnexus_merged.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('config', config_df, format='table')
    store.put('parts',parts_df, format='table')
    store.put('hits',hits_df, format='table')
    store.put('sns_response',sns_response_df, format='table')
    store.put('sns_position',sns_position_df, format='table')
    store.put('waveforms',waveform_df, format='table')