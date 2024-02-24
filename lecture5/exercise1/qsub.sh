#!/bin/bash
#SBATCH -p nvltv # partition (query with sinfo)
#SBATCH -o job-%j.out # stdout and stderr is written to this file
#SBATCH -N 1 # 1 node
#SBATCH -t 00:05:00 # maximal run time of the job
module load cuda

if [ -f time.dat ]; then
    rm time.dat
fi

for i in $(seq 1 10); do
    ./matvec i >tmp
    cat tmp | grep "time" | awk '{print '$i', "   ", $3}' >>time.dat
done
