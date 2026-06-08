# This script will convert the track reco files to a graph for ML training. 
# This is for using with a slurm script

# This notebook produces track objects by looping over the signal and background samples in segements

import pandas as pd
import numpy as np
import math
import glob
import sys
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.cluster import DBSCAN

# ------------------------------------------------------------------------------
# Get a subset of the total files to iterate over
def get_file_chunk(path, splitsize, chunk):
    
    files = sorted(glob.glob(path))
    total_files = len(files)
    
    chunk_size = math.ceil(total_files / splitsize)
    
    start = (chunk - 1) * chunk_size
    end = start + chunk_size
    
    return files[start:end]
# ------------------------------------------------------------------------------
class EventDataset(Dataset):
    def __init__(self, event_dfs_list, labels, voxel_size, spatial_shape):
        self.event_dfs_list = event_dfs_list
        self.labels = labels
        self.voxel_size = voxel_size
        self.spatial_shape = spatial_shape

        print("Voxel Size:", self.voxel_size, "mm", "| Voxel Grid Size:", self.spatial_shape, "| Total Events:", len(self.labels), "\n")

    def __getitem__(self, idx):
        
        # Grab the df for index 
        event_df = self.event_dfs_list[idx]
        coords, feats = voxelize_event(event_df, self.voxel_size)
        label = self.labels[idx]
        
        event_id = event_df.event_id.iloc[0] 
        subType = event_df['subType'].iloc[0]
        
        meta = {"event_id": event_id, "subType": subType}
        
        # Convert to torch objects
        coords = torch.from_numpy(coords).int()
        feats  = torch.from_numpy(feats).float()
        label  = torch.tensor(self.labels[idx])
        
        return coords, feats, label, meta

    def __len__(self):
        return len(self.event_dfs_list)
# ------------------------------------------------------------------------------
# Function to voxelize the event
# voxel size in mm
def process_voxel_event(event_df, voxel_size):

    # Convert the coorinates into integers
    z_int = (event_df['z'].values // voxel_size).astype(np.int32)
    y_int = (event_df['y'].values // voxel_size).astype(np.int32)
    x_int = (event_df['x'].values // voxel_size).astype(np.int32)

    group_df = pd.DataFrame({
        'event_id': event_df['event_id'].values,
        'z': z_int,
        'y': y_int,
        'x': x_int,
        'energy': event_df['energy'].values.astype(np.float32),
        'group_id': event_df['group_id'].values,
        'Type': event_df['Type'].values,
        'subType': event_df['subType'].values,
        'label': event_df['label'].values
    })
    
    voxel_event = group_df.groupby(['group_id', 'event_id', 'z', 'y', 'x'], as_index=False, sort=False).agg({
        'energy': 'sum',
        'Type': 'first',
        'subType': 'first',
        'label': 'first'
    })

    
    return voxel_event
# ------------------------------------------------------------------------------
def VoxelizeEventParallel(df, voxel_size, n_jobs=60):
    # Split by event_id - this is the unit of parallelization
    event_groups = df.groupby("event_id")
    
    # Process events in parallel across 60 cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_voxel_event)(group, voxel_size) 
        for _, group in event_groups
    )
    
    # Recombine
    return pd.concat(results, ignore_index=True)

# ------------------------------------------------------------------------------
def voxelize_event(event_df, VOXEL_SIZE):
    
    # Convert the coorinates into integers
    z_int = (event_df['z'].values // VOXEL_SIZE).astype(np.int32)
    y_int = (event_df['y'].values // VOXEL_SIZE).astype(np.int32)
    x_int = (event_df['x'].values // VOXEL_SIZE).astype(np.int32)

    group_df = pd.DataFrame({
        'event_id': event_df['event_id'].values,
        'z': z_int,
        'y': y_int,
        'x': x_int,
        'energy': event_df['energy'].values.astype(np.float32),
        'group_id': event_df['group_id'].values,
        'Type': event_df['Type'].values,
        'subType': event_df['subType'].values,
        'label': event_df['label'].values
    })
    
    voxel_df = group_df.groupby(['group_id', 'event_id', 'z', 'y', 'x'], as_index=False, sort=False).agg({
        'energy': 'sum',
        'Type': 'first',
        'subType': 'first',
        'label': 'first'
    })

    return voxel_df
# ------------------------------------------------------------------------------
# Creates a list of dataframes, one for each event, baed on test-train split
def make_event_df_list(df, event_ids):
    grouped = df.groupby('event_id')
    
    # Gets the whole dataframe for each eventid
    event_dfs_list = [grouped.get_group(eid).copy() for eid in event_ids]
    
    # Gets the labels for each event in the df
    labels = [g['label'].iloc[0] for g in event_dfs_list]
    return event_dfs_list, labels
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def GetSpatialShape(df, VOXEL_SIZE):
    # Estimate the max grid size based on the voxel size used
    max_z = max(df['z'])
    max_y = max(df['y'])
    max_x = max(df['x'])

    # Convert to voxel units and add a buffer
    global_max_coords = np.array([max_z, max_y, max_x])
    spatial_shape = np.ceil(global_max_coords / VOXEL_SIZE).astype(int) + 1

    # Round up to a multiple of 8 or 16
    # This ensures that stride-2 layers divide cleanly
    input_data_shape = [((s + 15) // 16) * 16 for s in spatial_shape]
    
    return input_data_shape
# ------------------------------------------------------------------------------
def MinMaxScale(df, label, var_min, var_max):
    # Min-Max scaling
    df[label] = (df[label] - var_min) / (var_max - var_min)
    return df
# ------------------------------------------------------------------------------
def ApplyScaling(df):
    # Normalize so that x,y,z are positive
    # df["z"] = df["z"]/6180.
    df["x"] = (df["x"]+500)
    df["y"] = (df["y"]+500)

    # Apply clipping to the energy, then scale it
    # df['energy'] = df['energy'].clip(upper=0.008)
    # df = MinMaxScale(df, "energy", 0, 0.008)

    # For energy, divide by the hits total event energy
    df_energy = df.groupby(["event", "Type"])["energy"].sum().reset_index(name="total_energy")
    df = df.merge(df_energy[["event", "total_energy"]], on="event")
    df["energy"] = df["energy"]/df["total_energy"]
    df = df[["event_id", "x", "y", "z", "energy", "Type", "subType"]]

    return df

# ------------------------------------------------------------------------------
# Load in the MC true for all events
def process_single_file(f, mode):
    
    filter_events = False
    if (len(event_list) != 0):
        filter_events = True
        
    # This is a list of selected events
    if (filter_events):
        event_list_values = event_list.event_id.values
    
    try:
        if (mode == "sophronia"):
            df = pd.read_hdf(f, "RECO/Events")
            df = df[["event", "X", "Y", "Z", "Ec"]]
        elif(mode == "nexus"):
            df = pd.read_hdf(f, "MC/particles")

        if (filter_events):
            df  = df[df.event_id.isin(event_list_values)] # filter events
        
        return df
    except Exception as e:
        print(f"Error in {f}: {e}")
        return None
# ------------------------------------------------------------------------------
def LoadFilesParallel(files,mode):
    
    # n_jobs=-1 uses all available cores (all 60)
    # prefer="threads" is good for I/O, but "processes" is better for pandas filtering
    results = Parallel(n_jobs=-1)(
        delayed(process_single_file)(f, mode) for f in files
    )
    
    # Filter out None results if any files failed to load
    results = [res for res in results if res is not None]
    
    return pd.concat(results)

# ------------------------------------------------------------------------------
# Get the data
def LoadData(f_sophronia):

    sophronia  = LoadFilesParallel(f_sophronia, "sophronia")
    nexus      = LoadFilesParallel(f_sophronia, "nexus")

    # Categrorize the events and return
    is_conv = nexus['final_proc'] == 'conv'
    nexus['Type'] = is_conv.groupby(nexus['event_id']).transform('any')
    nexus['Type'] = nexus['Type'].map({True: 'signal', False: 'background'})
    nexus = nexus[["event_id", "Type"]]
    nexus = nexus.drop_duplicates(subset=['event_id'])
    
    # Merge the true info into the sophronia files, then reformat
    sophronia = sophronia.merge(nexus, left_on="event", right_on="event_id")
    sophronia = sophronia[["event", "X", "Y", "Z", "Ec", "Type"]]
    sophronia.rename(columns={"event": "event_id", "X": "x", "Y": "y", "Z": "z", "Ec": "energy"}, inplace=True)
    sophronia["subType"] = sophronia["Type"]
    
    return sophronia
# ------------------------------------------------------------------------------
def GroupHits_single_event(event_id, event_df):

    threshold = np.sqrt(3)
    
    coords = event_df[["x", "y", "z"]].values
    
    # Apply DBSCAN
    db = DBSCAN(eps=threshold, min_samples=3).fit(coords)
    
    # We return the original DF with the new labels
    event_df = event_df.copy()
    event_df["group_id"] = db.labels_.astype(int)
    
    return event_df
# ------------------------------------------------------------------------------
# Since DB Scan can be slow, we can paralellize it across many CPUs
def GroupHitsParallel(df, n_jobs=60):

    # Split the dataframe by event_id
    event_groups = df.groupby("event_id")
    
    # Parallelize the DBSCAN
    results = Parallel(n_jobs=n_jobs)(
        delayed(GroupHits_single_event)(eid, group) for eid, group in event_groups
    )
    
    # Recombine into a single DataFrame
    return pd.concat(results, ignore_index=True)
# ------------------------------------------------------------------------------
# This function adds a column for the group id that is  unique across events
def MakeGlobalGroupIDs(df):

    # Get the max group ID of each event, shift them, and cumulative sum.
    event_maxes = df.groupby('event_id', sort=False)['group_id'].max() + 1
    event_offsets = event_maxes.shift(1, fill_value=0).cumsum()
    
    # Map those offsets back to the main dataframe
    df['group_id_offset'] = df['event_id'].map(event_offsets)
    
    # Create the final unique ID
    df['unique_group_id'] = df['group_id'] + df['group_id_offset']
    
    return df.drop(columns=['group_id_offset'])
# ------------------------------------------------------------------------------


basepath = "/media/argon/HardDrive_8TB/Krishan/Productions/NEXT100_Tl208_Port1a/"
listin=f"../eventlists/ATPC_1bar_5percent_highstats.csv"

VOXEL_SIZE=20.0

# The job id

# Job id needs to range from 1 to splitsize
jobid = int(sys.argv[1])
splitsize=int(sys.argv[2])

print("JobID/splitsize:", jobid, "/", splitsize)


filter_events=False

event_list = []

if filter_events:
    event_list = pd.read_csv(listin);


# Get all file names for each event category
f_sophronia   = get_file_chunk(f"{basepath}/sophronia/*.h5",  splitsize, jobid)

print("Num files loaded", len(f_sophronia))

print("Loading data")

# Load the data
df = LoadData(f_sophronia)
print(df)

# Apply scaling
df = ApplyScaling(df)
print("df after scaling")
print(df)

# Get the spatial shape
input_data_shape = GetSpatialShape(df, VOXEL_SIZE)

# Type: "signal" or "background"
df['label'] = (df['Type'] == "signal").astype(int)

# Apply grouping
# print("Grouping Dataframe")
# df = GroupHitsParallel(df)

# Apply a voxelization
print("Voxelizing Dataframe")
df = VoxelizeEventParallel(df, VOXEL_SIZE)

# print("Making Global Group IDs")
# df = MakeGlobalGroupIDs(df)
print(df)

# Event-level labels
event_labels = df.groupby('event_id')['label'].first()
event_ids    = event_labels.index.values
event_y      = event_labels.values

# Split 70/20/10
# We are splitting the dataset by the event ids
ev_tmp, ev_test, y_tmp, y_test   = train_test_split(event_ids, event_y, test_size=0.10, stratify=event_y, random_state=jobid)
ev_train, ev_val, y_train, y_val = train_test_split(ev_tmp, y_tmp, test_size=2/9, stratify=y_tmp, random_state=jobid)

train_events_df_list, train_labels = make_event_df_list(df, ev_train)
val_events_df_list,   val_labels   = make_event_df_list(df, ev_val)
test_events_df_list,  test_labels  = make_event_df_list(df, ev_test)

print("\nTrain Events:", len(train_events_df_list))
print("Val Events:  ", len(val_events_df_list))
print("Test Events: ", len(test_events_df_list), "\n")

if (len(event_list) != 0):
        print("Saving graph to ", f'{basepath}/CNN_files_MLP/ATPC_CNN_chunk_[train/val/test]_{jobid}.pt')
        torch.save(train_events_df_list,   f'{basepath}/CNN_files_MLP/ATPC_CNN_chunk_train_{jobid}.pt')
        torch.save(val_events_df_list,     f'{basepath}/CNN_files_MLP/ATPC_CNN_chunk_val_{jobid}.pt')
        torch.save(test_events_df_list,    f'{basepath}/CNN_files_MLP/ATPC_CNN_chunk_test_{jobid}.pt')
        
else:
    print("Saving dataset to ", f'{basepath}/CNN_files/ATPC_CNN_chunk_[train/val/test]_{jobid}.pt')
    torch.save(train_events_df_list,   f'{basepath}/CNN_files/ATPC_CNN_chunk_train_{jobid}.pt')
    torch.save(val_events_df_list,     f'{basepath}/CNN_files/ATPC_CNN_chunk_val_{jobid}.pt')
    torch.save(test_events_df_list,    f'{basepath}/CNN_files/ATPC_CNN_chunk_test_{jobid}.pt')