#!/bin/bash
#SBATCH -J NEXT100_LPR_LowTh # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o NEXT100_LPR_LowTh_%A_%a.out # Standard output
#SBATCH -e NEXT100_LPR_LowTh_%A_%a.err # Standard error

start=`date +%s`

# Set the configurable variables
JOBNAME="NEXT100_LPR_LowTh"

# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

# Get the row in the file to get the input
# Set the path to your text file
file_path="file_paths.txt"

inputfile=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$file_path")
echo "Input file: $inputfile"

# Go into the running area
mkdir -p job_output/job${SLURM_ARRAY_TASK_ID}
cd job_output/job${SLURM_ARRAY_TASK_ID}

# Python
echo "Running Script" 
python /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/collect_lowTh.py ${inputfile}

mv LowTh.h5 /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/job/job_output/$(basename "$inputfile" .h5)_lowTh.h5

rm -r /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/job/job_output/job${SLURM_ARRAY_TASK_ID}

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 