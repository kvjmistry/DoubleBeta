#!/bin/bash
#SBATCH -J DEMO_Kr # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o DEMO_Kr_%A_%a.out # Standard output
#SBATCH -e DEMO_Kr_%A_%a.err # Standard error

start=`date +%s`

# Set the configurable variables
JOBNAME="Kr"
N_EVENTS=1
CONFIG=DEMOPP_fullKr.config.mac
INIT=DEMOPP_fullKr.init.mac

# Setup nexus and run
echo "Setting up GARFNEXUS"
source /home/argon/Projects/Krishan/GarfieldNexus/nexus/setup_cluster.sh
source /home/argon/Projects/Krishan/IC/setup_IC.sh

mkdir -p /media/argon/HDD_8tb/Krishan/DEMO/nexus/${JOBNAME}/job${SLURM_ARRAY_TASK_ID}
cd       /media/argon/HDD_8tb/Krishan/DEMO/nexus/${JOBNAME}/job${SLURM_ARRAY_TASK_ID}

cp /home/argon/Projects/Krishan/GarfieldNexus/nexus/macros/${INIT} .
cp /home/argon/Projects/Krishan/GarfieldNexus/nexus/macros/${CONFIG} .
cp /home/argon/Projects/Krishan/DoubleBeta/GarfNexus/scripts/slim_files.py .

sed -i "s#.*RegisterMacro.*#/nexus/RegisterMacro ${CONFIG}#" ${INIT}
sed -i "s#.*specific_vertex.*#/Geometry/NextDemo/specific_vertex 0 0 ${SLURM_ARRAY_TASK_ID} mm#" ${CONFIG}
sed -i "s#.*output_file.*#/nexus/persistency/output_file DEMOpp_Kr_Z${SLURM_ARRAY_TASK_ID}mm#" ${CONFIG}

# Replace the seed in the file
SEED=$((${N_EVENTS}*(${SLURM_ARRAY_TASK_ID} - 1) + ${N_EVENTS}))
echo "The seed number is: ${SEED}"
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}

# Launch nexus
nexus ${INIT} -n ${N_EVENTS}

# Compress the file
python3 slim_files.py DEMOpp_Kr_Z${SLURM_ARRAY_TASK_ID}mm

# Clean up
rm DEMOpp_Kr_Z${SLURM_ARRAY_TASK_ID}mm.h5
rm ${INIT}
rm ${CONFIG}
rm slim_files.py
rm DEGRAD.OUT
rm degrad_print.txt
rm conditions_Degrad.txt

ls -ltrh 

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 