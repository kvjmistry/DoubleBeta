#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

echo "JOBID $1 running on `whoami`@`hostname`"
start=`date +%s`

SCRIPT=$3
echo "Script name is: ${SCRIPT}"
start=`date +%s`

# Setup nexus
echo "Setting Up NEXUS" 
source /software/garfnexus/setup_nexus.sh

# Set the configurable variables
N_EVENTS=50
CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

echo "N_EVENTS: ${N_EVENTS}"

SEED=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
echo "The seed number is: ${SEED}" 

# Change the config in the files
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}
sed -i "s#.*output_file.*#/nexus/persistency/output_file ${JOBNAME}#" ${CONFIG}

# Print out the config and init files
cat ${INIT}
cat ${CONFIG}

# NEXUS
echo "Running NEXUS" 
alias python="python3"
nexus -n $N_EVENTS ${INIT}
python3 slim_files.py ${JOBNAME}

rm "${JOBNAME}.h5"

ls -ltrh

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 
