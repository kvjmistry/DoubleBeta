#!/bin/bash
#SBATCH -J FILTER # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/FILTER_%A_%a.out # Standard output
#SBATCH -e log/FILTER_%A_%a.err # Standard error
#SBATCH --array=1-326

start=`date +%s`

# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

RUN_NUMBER=13850 # 326
# RUN_NUMBER=14180 # 644
# RUN_NUMBER=14498 # 343
# RUN_NUMBER=14780 # 568%15

mkdir -p /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filteredC/${RUN_NUMBER}/
cd       /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filteredC/${RUN_NUMBER}/


input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/DoubleBeta/Alphas/eventlists/run_${RUN_NUMBER}_filtfiles.txt)
echo "Input File: $input_file"


python /home/argon/Projects/Krishan/DoubleBeta/Alphas/notebooks/CalibrateDataframe.py $input_file ${RUN_NUMBER}


echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 