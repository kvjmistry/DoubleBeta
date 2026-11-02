#!/bin/bash
#SBATCH -J ESMERALDA # A single job name for the array
#SBATCH --nodes=1
#SBATCH --mem 4000 # Memory request (6Gb)
#SBATCH -t 0-1:00 # Maximum execution time (D-HH:MM)
#SBATCH -o log/ESMERALDA_%A_%a.out # Standard output
#SBATCH -e log/ESMERALDA_%A_%a.err # Standard error
#SBATCH --array=1-9381

start=`date +%s`


# Setup nexus and run
echo "Setting up IC"
source /home/argon/Projects/Krishan/IC/setup_IC.sh

mkdir -p /media/argon/HardDrive_8TB/Krishan/Productions/NEXT100_Th228_Port1a_FSPECT/4bar/esmeralda/
cd       /media/argon/HardDrive_8TB/Krishan/Productions/NEXT100_Th228_Port1a_FSPECT/4bar/esmeralda/


input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/eventlists/NEXT100_Th228_Port1a_FSPECT_sophronia.txt | tr -d '\r')
echo "Input File: $input_file"

# Make the new filename
base=$(basename "$input_file" _sophronia.h5)
outfile="${base}_esmeralda.h5"

echo "$outfile"

city esmeralda /home/argon/Projects/Krishan/DoubleBeta/Bremsstrahlung/config/esmeralda.conf -i $input_file -o ${outfile}


# ls -ltrh 

echo "FINISHED....EXITING"

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds 