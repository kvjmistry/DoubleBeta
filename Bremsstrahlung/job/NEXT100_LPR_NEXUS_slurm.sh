#!/bin/bash
#SBATCH -J NEXT100_NEXUS # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o NEXT100_LPR_%A_%a.out # Standard output
#SBATCH -e NEXT100_LPR_%A_%a.err # Standard error

start=`date +%s`

# Set the configurable variables
MODE="GS"
INIT=NEXT100_${MODE}.init.mac
CONFIG=NEXT100_${MODE}.config.mac

# Setup nexus and run
echo "Setting up NEXUS"
source /home/argon/Projects/Krishan/nexus/setup_cluster.sh
source /home/argon/Projects/Krishan/IC/setup_IC.sh

# Go into the running area
mkdir -p /media/argon/HDD_8tb/Krishan/Bremsstrahlung/${MODE}/job${SLURM_ARRAY_TASK_ID}
cd /media/argon/HDD_8tb/Krishan/Bremsstrahlung/${MODE}/job${SLURM_ARRAY_TASK_ID}

cp /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/config/${INIT} .
cp /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/config/${CONFIG} .
cp /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/config/compress_files.py .

SEED=$((${SLURM_ARRAY_TASK_ID}+1))
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}

# Python
echo "Running Software" 
nexus ${INIT} -n 50000
python3 compress_files.py Next100_Tl208_Port1a_${MODE}

rm Next100_Tl208_Port1a_${MODE}.h5
rm *.mac
rm *.py

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 