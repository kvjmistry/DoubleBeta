#!/bin/bash
#SBATCH -J FILTER # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o FILTER_%A_%a.out # Standard output
#SBATCH -e FILTER_%A_%a.err # Standard error

start=`date +%s`

# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

# RUN_NUMBER=13850
RUN_NUMBER=14180

mkdir -p /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filtered/${RUN_NUMBER}/
cd       /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filtered/${RUN_NUMBER}/


input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/DoubleBeta/Alphas/eventlists/run_${RUN_NUMBER}_files.txt)
echo "Input File: $input_file"


python /home/argon/Projects/Krishan/DoubleBeta/Alphas/notebooks/StudyWaveforms.py $input_file ${RUN_NUMBER}


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 