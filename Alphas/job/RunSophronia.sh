#!/bin/bash
#SBATCH -J SOPHRONIA # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/SOPHRONIA_%A_%a.out # Standard output
#SBATCH -e log/SOPHRONIA_%A_%a.err # Standard error
#SBATCH --array=1-343

start=`date +%s`

# RUN_NUMBER=13850 # 326
# RUN_NUMBER=14180 # 644
RUN_NUMBER=14498 # 343
# RUN_NUMBER=14780 # 568%15
# RUN_NUMBER=16156 # 829%15


# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

mkdir -p /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/sophronia/${RUN_NUMBER}/
cd       /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/sophronia/${RUN_NUMBER}/


input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/DoubleBeta/Alphas/eventlists/run_${RUN_NUMBER}_irene.txt | tr -d '\r')
echo "Input File: $input_file"

# Make the new filename
base=$(basename "$input_file" _irene.h5)
outfile="${base}_sophronia.h5"

echo "$outfile"

city sophronia /home/argon/Projects/Krishan/DoubleBeta/Alphas/config/sophronia_${RUN_NUMBER}.conf -i $input_file -o ${outfile}


# ls -ltrh 

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 