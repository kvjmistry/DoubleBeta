# RUN_NUMBER=13850
RUN_NUMBER=14180
FILEPATH=/media/argon/HardDrive_8TB/Krishan/NEXT100Data/alpha/filtered/
mkdir -p ../eventlists/
for f in $(ls ${FILEPATH}/${RUN_NUMBER}/); do realpath ${FILEPATH}/${RUN_NUMBER}/$f >> ../eventlists/run_${RUN_NUMBER}_filtfiles.txt; done