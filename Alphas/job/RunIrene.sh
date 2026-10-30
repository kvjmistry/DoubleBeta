#!/bin/bash
#SBATCH -J IRENE # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o IRENE_%A_%a.out # Standard output
#SBATCH -e IRENE_%A_%a.err # Standard error

start=`date +%s`

# RUN_NUMBER=13850 # 326
# RUN_NUMBER=14180 # 644
RUN_NUMBER=14498 # 343
# RUN_NUMBER=14780 # 568%15
# RUN_NUMBER=16156 # 829%15


# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

mkdir -p /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/irene/${run_number}/
cd       /media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/irene/${run_number}/


input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/DoubleBeta/Alphas/eventlists/run_${run_number}_files.txt)
echo "Input File: $input_file"

# Make the new filename
base=$(basename "$filepath" .waveforms.h5)
outfile="${base}_irene.h5"

echo "$outfile"

city irene /home/argon/Projects/Krishan/DoubleBeta/Alphas/config/irene_${run_number}.conf -i $input_file -o ${outfile}


ls -ltrh 

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 