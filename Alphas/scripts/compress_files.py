# Python script to slim nexus files
import sys
import pandas as pd

# python3 <basename>

#config = pd.read_hdf(sys.argv[1], key = 'MC/configuration')
#print(config)

parts = pd.read_hdf(sys.argv[1], key = 'MC/particles')
parts = parts[parts.primary == 1]
parts = parts[(parts.initial_x > -500) & (parts.initial_x < 500) & (parts.initial_y > -500) & (parts.initial_y < 500)] # skim off events just outside the bin range
parts = parts[["event_id", "initial_x", "initial_y"]]
parts["initial_x"] = parts["initial_x"].astype("float16")
parts["initial_y"] = parts["initial_y"].astype("float16")
print(parts)

# Same with the hits
#hits = pd.read_hdf(sys.argv[1], key = 'MC/hits')
#print(hits)

# Same with the sensor hits
#sns_response = pd.read_hdf(sys.argv[1], key = 'MC/sns_response')
#print(sns_response)

# Same with the sensor positions
#sns_positions = pd.read_hdf(sys.argv[1], key = 'MC/sns_positions')
#print(sns_positions)


# Open the HDF5 file in write mode
with pd.HDFStore(f"{sys.argv[1]}", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    #store.put('MC/configuration', config, format='table')
    store.put('MC/particles',parts, format='table')
    #store.put('MC/hits',hits, format='table')
    #store.put('MC/sns_response',sns_response, format='table')
    #store.put('MC/sns_positions',sns_positions, format='table')
