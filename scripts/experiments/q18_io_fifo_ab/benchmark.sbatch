#!/bin/bash
#SBATCH --job-name=q18-io-fifo-ab
#SBATCH --partition=nvidia-part
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --mem=850G
#SBATCH --time=00:55:00
#SBATCH --chdir=/home/wmalpica/repos/sirius
set -euo pipefail
source /etc/profile
module load slurm
root=${1:?Pass a fresh absolute output directory}
a_ref=${2:?Pass the A ref}
b_ref=${3:?Pass the B ref}
# The submission instructions create this directory for Slurm stdout.
mkdir "$root/scripts"
cp scripts/experiments/q18_io_fifo_ab/{run-ab.sh,driver.py,check-placement.py,analyze-queue.py} "$root/scripts/"
cp "$root/scripts/"*.py "$root/"
scontrol show job "$SLURM_JOB_ID" > "$root/job-$SLURM_JOB_ID.txt"
srun --exact --ntasks=1 --cpus-per-task=72 --distribution=block:block --cpu-bind=none --gres=gpu:b200:2 bash "$root/scripts/run-ab.sh" "$root" "$a_ref" "$b_ref"
