#!/bin/bash
#SBATCH -J DIFFUSION # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 2000 # Memory request (6Gb)
#SBATCH -t 0-6:00 # Maximum execution time (D-HH:MM)
#SBATCH -o DIFFUSION_%A_%a.out # Standard output
#SBATCH -e DIFFUSION_%A_%a.err # Standard error

# To run the job you run sbatch in the terminal:
# sbatch --array=1-10 <this script name>.sh
# The array is the range of jobs to run e.g. this runs 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# Copy the files over
# Create the directory

# Set the configurable variables
JOBNAME="DIFFUSION"
FILES_PER_JOB=1
N_EVENTS=50

MODE="eminus"
CONFIG=XeSphere_$MODE.config.mac 
INIT=XeSphere_$MODE.init.mac


#Create the working area
cd /media/argon/HDD_8tb/Krishan
mkdir -p $JOBNAME/$MODE/jobid_"${SLURM_ARRAY_TASK_ID}"
cd $JOBNAME/$MODE/jobid_"${SLURM_ARRAY_TASK_ID}"

# ---
cp /home/argon/Projects/Krishan/DoubleBeta/job/${CONFIG} .
cp /home/argon/Projects/Krishan/DoubleBeta/job/${INIT} .

echo "Initialising environment" 
start=`date +%s`


# Setup nexus and run
echo "Setting Up NEXUS" 
source ~/Projects/Krishan/nexus/setup_cluster.sh

# Also setup IC
source /home/argon/Software/IC/setup_IC.sh


for i in $(eval echo "{1..${FILES_PER_JOB}}"); do

    # Replace the seed in the file	
    SEED=$((${N_EVENTS}*${FILES_PER_JOB}*(${SLURM_ARRAY_TASK_ID} - 1) + ${N_EVENTS}*${i}))
    echo "The seed number is: ${SEED}" 
    sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
    sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}

    # NEXUS
    echo "Running NEXUS" 
    nexus -n $N_EVENTS ${INIT} 

    # Diffusion Script
    echo "Running Diffusion"
    python /home/argon/Projects/Krishan/DoubleBeta/notebooks/DiffuseData.py

    mv xesphere.h5 xesphere_1bar_${MODE}_${SLURM_ARRAY_TASK_ID}.h5

    echo; echo; echo;
done

rm 0nuBB2.next.h5

# Remove the config files if not the first jobid
if [ ${SLURM_ARRAY_TASK_ID} -ne 1 ]; then
    rm -v *.conf 
    rm -v *.mac 
fi

echo "FINISHED....EXITING" 
end=`date +%s`
runtime=$((end-start))
echo "$runtime s" 