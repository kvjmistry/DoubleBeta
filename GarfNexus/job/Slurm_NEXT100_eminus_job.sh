#!/bin/bash
#SBATCH -J GN # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-24:00 # Maximum execution time (D-HH:MM)
#SBATCH -o GN_%A_%a.out # Standard output
#SBATCH -e GN_%A_%a.err # Standard error

start=`date +%s`
 
# Set the configurable variables
JOBNAME="NEXT100_eminus"
# Set the configurable variables
N_EVENTS=50
CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

# Create the directory
cd /media/argon/HDD_8tb/
mkdir -p $JOBNAME/jobid_"${SLURM_ARRAY_TASK_ID}"
cd $JOBNAME/jobid_"${SLURM_ARRAY_TASK_ID}"

cp /home/argon/Projects/Krishan/DoubleBeta/GarfNexus/config/* .
cp /home/argon/Projects/Krishan/DoubleBeta/GarfNexus/scripts/slim_files.py .

# Setup nexus and run
echo "Setting Up NEXUS" 2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt
source /home/argon/Projects/Krishan/GarfieldNexus/nexus/setup_cluster.sh
source /home/argon/Projects/Krishan/IC/setup_IC.sh

# Calculate the unique seed number  
SEED=$((${N_EVENTS}*(${SLURM_ARRAY_TASK_ID} - 1) + ${N_EVENTS}))
echo "The seed number is: ${SEED}" 2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt

# Change the config in the files
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}
sed -i "s#.*output_file.*#/nexus/persistency/output_file ${JOBNAME}#" ${CONFIG}

# Print out the config and init files
cat ${INIT}  2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt
cat ${CONFIG}  2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt

# NEXUS
echo "Running NEXUS" 2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt
nexus -n $N_EVENTS ${INIT}   2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt
python3 slim_files.py ${JOBNAME}  2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt

rm "${JOBNAME}.h5"

ls -ltrh  2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt

echo; echo; echo;

echo "FINISHED....EXITING" 2>&1 | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds | tee -a log_nexus_"${SLURM_ARRAY_TASK_ID}".txt