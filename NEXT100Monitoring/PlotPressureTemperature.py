import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

def PlotPressure(ax1,ax2, varname, df_mean,df, col):

    ax1.plot(df_mean.index, df_mean[f'{varname}Delta']*1000, marker='o', linestyle='-', color = col)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Pressure Change [mbar]')

    ax1_copy = ax1.twinx()
    ax1_copy.plot(df_mean.index, df_mean[varname], color = "k")
    ax1_copy.tick_params(axis='y')
    ax1_copy.set_ylabel(f'Pressure [bar]')
    ax1_copy.grid()
    ax1_copy.tick_params(axis='y') 
    ax1.set_title(f"{varname} hourly mean")

    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    fig.tight_layout()

    ax2.plot(df.index, df[f'{varname}'], linestyle='-', color = col, alpha = 0.3, label = "Raw")
    # ax2.plot(df_mean.index, df_mean[f'{varname}Delta'], marker='o', linestyle='-', color = col)
    ax2.plot(df_mean.index, df_mean[f'{varname}'], marker='o', linestyle='-', color = "k", label = "Hourly Mean")
    ax2.set_xlabel('Date')
    ax2.set_ylabel(f'Pressure [mbar]')
    ax2.set_title(f"{varname}")

    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    ax2.legend()


def LoadData(paths):

    dfs = []

    # Loop over each file path
    for file_path in paths:
        # Read the file into a DataFrame
        df = pd.read_csv(file_path, sep='\t', skiprows=2, decimal=',', encoding='ISO-8859-1')

        for col in df.columns:
            if (col != "Date" and col != "Hour"):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)


    # Convert the 'Date' and 'Hour' columns to a single datetime column
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Hour'],
        format='%d/%m/%Y %H:%M:%S',  # Adjust format according to your data
        dayfirst=True
    )

    # Display the final DataFrame
    print(df.head())

    df.set_index('Datetime', inplace=True)

    df = df.select_dtypes(include=[np.number])

    df_mean = df.copy()
    df_mean = df_mean.resample('h').mean()

    # Display the final DataFrame
    print(df_mean.head())

    return df, df_mean


def PlotDICE(ax1, ax2, df, varname, title):
     
    all_vals = np.array([])
    mean_vals = np.array([])

    for i in range(0,55,1):
        all_vals = np.append(all_vals,df[f'DICE{i}{varname}'])
        mean_vals = np.append(mean_vals,np.mean(df[f'DICE{i}{varname}']))

    median_of_means = np.median(mean_vals)
    mad = np.median(np.abs(mean_vals - median_of_means))
    

    for i in range(0,55,1):
        if (mean_vals[i] > median_of_means+2*mad or mean_vals[i] < median_of_means-2*mad):
            ax1.plot(df.index, df[f'DICE{i}{varname}'], label=f'DICE{i}{varname}', linestyle='-')
        else:
            ax1.plot(df.index, df[f'DICE{i}{varname}'], linestyle='-', color = "k", alpha=0.2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'DICE {title}')
    ax1.grid(True)
    ax1.legend(ncol = 2)

    if ("Temp" in title):
        ax1.set_ylim(np.median(all_vals)-1, np.median(all_vals)+1)
    if ("Volt" in title):
        ax1.set_ylim(0, 60)

    if ("Curr" in title):
        ax1.set_ylim(0, 1.6e-5)

    for label in ax1.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    for index, i in enumerate(mean_vals):
        if (mean_vals[index] > median_of_means+2*mad or mean_vals[index] < median_of_means-2*mad):
            ax2.scatter(index, i, label = f'DICE{index}{varname}')
        else:
            ax2.scatter(index, i, color = "k", alpha = 0.2)
        
    ax2.axhline(median_of_means+3*mad, linestyle = "--", color = "Teal")
    ax2.axhline(median_of_means-3*mad, linestyle = "--", color = "Teal")
    ax2.axhline(median_of_means+2*mad, linestyle = "--", color = "DarkOrange")
    ax2.axhline(median_of_means-2*mad, linestyle = "--", color = "DarkOrange")
    ax2.axhline(median_of_means, linestyle = "--", color = "k")

    ax2.set_xlabel("DICE ID")
    ax2.set_ylabel(f"Mean {title}")
    ax2.set_ylim(median_of_means-5*mad, median_of_means+5*mad)
    ax2.legend(ncol = 2)

    print("Median: ",median_of_means)
    print("Mad: ",mad)
    for index,u in enumerate(mean_vals):
         if (u > median_of_means+3*mad or u < median_of_means-3*mad):
              print(f"DICE {index} out of {title} range with val {u}" )

# Define the directory containing the text files
Gas_files = '/Users/mistryk2/Desktop/Pressure_Data/'

# Define the directory containing the text files
DICE_files = '/Users/mistryk2/Desktop/Dice_Temps/'

# path = r'C:\Users\next\DIPC Dropbox\NextElec Zulo\Nextelec\- Slow Control NEXT-100\SC Reports'
# Gas_files = fr'{path}\GAS\Data\\'
# DICE_files = fr'{path}\TP\Data\\'

# Number of days to look back
N_days = 6

# Get the pressure files from the previous 5 days
pressure_paths = sorted(glob.glob(f"{Gas_files}/*Pressure*.txt"), reverse = True)[0:N_days]

GasPressure, GasPressureMean = LoadData(pressure_paths)

GasPressureMean["PG3Delta"] = GasPressureMean["PG3"] - GasPressureMean["PG3"].iloc[0]
GasPressureMean["PG6Delta"] = GasPressureMean["PG6"] - GasPressureMean["PG6"].iloc[0]


# Load in the DICE
# Use glob to get all text files in the directory
DICE_paths = sorted(glob.glob(f"{DICE_files}/TP_SiPM_BS*.txt"), reverse=True)[0:N_days]

# Initialize an empty list to store individual DataFrames
DICE_Temps, DICE_TempsMean = LoadData(DICE_paths)


# --------------------------------------------------
# Plot the pressures
fig, axes = plt.subplots(2,2, figsize=(15, 8))
ax1, ax2, ax3, ax4 = axes.flatten()
PlotPressure(ax1, ax2, "PG3", GasPressureMean, GasPressure, "Teal")
PlotPressure(ax3, ax4, "PG6", GasPressureMean, GasPressure, "DarkOrange")

# --------------------------------------------------
# Plot the DICE Temps
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15, 8))
PlotDICE(ax1, ax2, DICE_TempsMean, "(ºC) T", "Temperature [Celcius]")

# Plot the DICE Voltages
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15, 8))
PlotDICE(ax1, ax2, DICE_TempsMean, "(V) V", "Voltage [Volts]")

# Plot the DICE Currents
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15, 8))
PlotDICE(ax1, ax2, DICE_TempsMean, "(A) I", "Current [A]")

# --------------------------------------------------
# Plot the ratio to DICE0
plt.figure(figsize=(10, 6))

Ratio = GasPressureMean['PG3']/DICE_TempsMean['DICE0(ºC) T']

plt.plot(DICE_TempsMean.index,  Ratio, color= "Teal", linewidth = 4)

plt.xlabel('Date')
plt.ylabel('Ratio PG3/DICE 0 Temp [bar/Celcius]')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(np.mean(Ratio)-0.1, np.mean(Ratio)+0.1)

plt.show()


