#!/bin/bash
#SBATCH -J PHOTOELECTRIC # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/PHOTOELECTRIC_%A_%a.out # Standard output
#SBATCH -e log/PHOTOELECTRIC_%A_%a.err # Standard error

start=`date +%s`

# Setup nexus and run
echo "Setting up nexus"
source /home/argon/Projects/Krishan/nexus/setup_cluster.sh
source /home/argon/Projects/Krishan/venv/bin/activate

JOBNAME="photoelectric"
N_EVENTS=5000000
CONFIG=NEXT100_S2_LT.config.mac
INIT=NEXT100_S2_LT.config.mac

# Create the directory
cd /home/argon/Projects/Krishan/DoubleBeta/Alphas/
mkdir -p $JOBNAME/jobid_"${SLURM_ARRAY_TASK_ID}"
cd $JOBNAME/jobid_"${SLURM_ARRAY_TASK_ID}"

cp /home/argon/Projects/Krishan/DoubleBeta/Alphas/config/*mac* .

# Calculate the unique seed number  
SEED=$((${N_EVENTS}*(${SLURM_ARRAY_TASK_ID} - 1) + ${N_EVENTS}))
echo "The seed number is: ${SEED}"

# Change the config in the files
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}
sed -i "s#.*output_file.*#/nexus/persistency/output_file ${JOBNAME}_${SLURM_ARRAY_TASK_ID}#" ${CONFIG}

# NEXUS
echo "Running NEXUS" 
nexus -n $N_EVENTS ${INIT}
python3 /home/argon/Projects/Krishan/DoubleBeta/Alphas/scripts/compress_files.py ${JOBNAME}_${SLURM_ARRAY_TASK_ID}.h5

rm *.mac

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 