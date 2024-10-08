#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

PRESSURE=$3
echo "The Pressure number is: ${PRESSURE}" 

echo "JOBID $1 running on `whoami`@`hostname`"
start=`date +%s`

# Setup nexus
echo "Setting Up NEXUS and IC" 
source /software/nexus/setup_nexus.sh
# source /software/IC/setup_IC.sh

# Set the configurable variables
N_EVENTS=10000
CONFIG=${JOBNAME}.config.mac
INIT=${JOBNAME}.init.mac

echo "N_EVENTS: ${N_EVENTS}"

SEED=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
echo "The seed number is: ${SEED}" 

# Change the config in the file
sed -i "s#.*pressure.*#/Geometry/Next100/pressure ${PRESSURE} bar#" ${CONFIG}
sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${SEED}#" ${CONFIG}

# Print out the config and init files
cat ${INIT}
cat ${CONFIG}

# NEXUS
echo "Running NEXUS" 
nexus -n $N_EVENTS ${INIT}
echo "Slimming file" 
python3 compress_files.py NEXUS_OUTPUT

ls -ltrh

echo "Removing nexus output"
rm NEXUS_OUTPUT.h5

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 