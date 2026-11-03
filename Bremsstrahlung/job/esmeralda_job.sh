#!/bin/bash

echo "Starting Job" 

JOBID=$1
echo "The JOBID number is: ${JOBID}" 

JOBNAME=$2
echo "The JOBNAME number is: ${JOBNAME}" 

RUN_NUMBER=$3
echo "The RUN_NUMBER is: ${RUN_NUMBER}" 

INFILE=$4
echo "The input file is: ${INFILE}" 

echo "JOBID ${JOBID} running on `whoami`@`hostname`"

start=`date +%s`

# Setup nexus
echo "Setting Up Software" 
source /software/nexus/setup_nexus.sh
source /software/IC/setup_IC.sh

echo "Printing esmeralda file"

# Update the run number
sed -i "s#.*run_number.*#run_number = {RUN_NUMBER}#" esmeralda.conf

cat esmeralda.conf

# Update the filename
fileout="${INFILE/sophronia/esmeralda}"

echo "Running esmeralda" 
city esmeralda esmeralda.conf -i $INFILE -o ${fileout}

rm *sophronia*

ls -ltrh

echo "Taring the h5 files"
tar -cvf ${JOBNAME}.tar *.h5

# Cleanup
rm *.h5
# rm *.mac
rm *.conf
# rm *.py

echo "FINISHED....EXITING" 

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 